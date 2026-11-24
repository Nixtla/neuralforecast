// Fused CPU multiplicative ES scan and its reverse-mode derivative.
// No approximation, detached states, or change to the ES-RNN recurrence.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cmath>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <type_traits>
#endif

using at::Tensor;

std::vector<Tensor> scan_forward(Tensor y, Tensor c, Tensor init1, Tensor init2) {
  TORCH_CHECK(y.device().is_cpu() && c.device().is_cpu() && init1.device().is_cpu() && init2.device().is_cpu(), "ES scan requires CPU tensors");
  TORCH_CHECK(y.dim()==2 && c.dim()==2 && init1.dim()==2 && init2.dim()==2, "ES scan requires matrices");
  const int64_t B=y.size(0), T=y.size(1), m=init1.size(1), n=init2.size(1);
  TORCH_CHECK(T>0 && c.size(0)==B && init1.size(0)==B && init2.size(0)==B, "ES scan shape mismatch");
  TORCH_CHECK(c.size(1)==1+(m>0)+(n>0) && (m>0 || n==0), "ES scan seasonal shape mismatch");
  TORCH_CHECK(y.is_contiguous() && c.is_contiguous() && init1.is_contiguous() && init2.is_contiguous(), "ES scan requires contiguous tensors");
  TORCH_CHECK(c.scalar_type()==y.scalar_type() && init1.scalar_type()==y.scalar_type() && init2.scalar_type()==y.scalar_type(), "ES scan dtype mismatch");
  auto levels=at::empty_like(y);
  auto s1=at::empty({B,m ? T+m : 0},y.options());
  auto s2=at::empty({B,n ? T+n : 0},y.options());
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "es_scan_forward", [&] {
    const auto *Y=y.data_ptr<scalar_t>(), *C=c.data_ptr<scalar_t>();
    const auto *I=init1.data_ptr<scalar_t>(), *J=init2.data_ptr<scalar_t>();
    auto *L=levels.data_ptr<scalar_t>(), *S=s1.data_ptr<scalar_t>(), *R=s2.data_ptr<scalar_t>();
    const int64_t nc=c.size(1), ns=m ? T+m : 0, nr=n ? T+n : 0;
    for(int64_t b=0;b<B;++b) {
      for(int64_t j=0;j<m;++j) S[b*ns+j]=I[b*m+j];
      for(int64_t j=0;j<n;++j) R[b*nr+j]=J[b*n+j];
      if(m) S[b*ns+m]=S[b*ns];
      if(n) R[b*nr+n]=R[b*nr];
      const scalar_t a=C[b*nc], g=m ? C[b*nc+1] : 0, h=n ? C[b*nc+2] : 0;
      L[b*T]=Y[b*T]/((m ? S[b*ns] : scalar_t(1))*(n ? R[b*nr] : scalar_t(1)));
      for(int64_t t=1;t<T;++t) {
        const scalar_t u=m ? S[b*ns+t] : scalar_t(1), v=n ? R[b*nr+t] : scalar_t(1), z=Y[b*T+t];
        const scalar_t l=a*z/(u*v)+(scalar_t(1)-a)*L[b*T+t-1];
        L[b*T+t]=l;
        if(m) S[b*ns+t+m]=g*z/(l*v)+(scalar_t(1)-g)*u;
        if(n) R[b*nr+t+n]=h*z/(l*u)+(scalar_t(1)-h)*v;
      }
    }
  });
  return {levels,s1,s2};
}

std::vector<Tensor> scan_backward(Tensor y, Tensor c, Tensor levels, Tensor s1, Tensor s2,
                                  Tensor gl, Tensor gs, Tensor gr, int64_t m, int64_t n) {
  // Gradient workspaces are private: never mutate the caller's output gradients.
  gl=gl.clone(); gs=gs.clone(); gr=gr.clone();
  auto dy=at::zeros_like(y), dc=at::zeros_like(c);
  auto di=at::zeros({y.size(0),m},y.options()), dj=at::zeros({y.size(0),n},y.options());
  const int64_t B=y.size(0), T=y.size(1), nc=c.size(1), ns=m ? T+m : 0, nr=n ? T+n : 0;
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "es_scan_backward", [&] {
    const auto *Y=y.data_ptr<scalar_t>(), *C=c.data_ptr<scalar_t>(), *L=levels.data_ptr<scalar_t>();
    const auto *S=s1.data_ptr<scalar_t>(), *R=s2.data_ptr<scalar_t>();
    auto *GL=gl.data_ptr<scalar_t>(), *GS=gs.data_ptr<scalar_t>(), *GR=gr.data_ptr<scalar_t>();
    auto *DY=dy.data_ptr<scalar_t>(), *DC=dc.data_ptr<scalar_t>();
    auto *DI=di.data_ptr<scalar_t>(), *DJ=dj.data_ptr<scalar_t>();
    for(int64_t b=0;b<B;++b) {
      const scalar_t a=C[b*nc], g=m ? C[b*nc+1] : 0, h=n ? C[b*nc+2] : 0;
      for(int64_t t=T-1;t>=1;--t) {
        const scalar_t z=Y[b*T+t], l=L[b*T+t];
        const scalar_t u=m ? S[b*ns+t] : scalar_t(1), v=n ? R[b*nr+t] : scalar_t(1);
        if(m) {
          const scalar_t grad=GS[b*ns+t+m], q=z/(l*v);
          DC[b*nc+1]+=grad*(q-u);
          DY[b*T+t]+=grad*g/(l*v);
          GL[b*T+t]-=grad*g*q/l;
          GS[b*ns+t]+=grad*(scalar_t(1)-g);
          if(n) GR[b*nr+t]-=grad*g*q/v;
        }
        if(n) {
          const scalar_t grad=GR[b*nr+t+n], q=z/(l*u);
          DC[b*nc+2]+=grad*(q-v);
          DY[b*T+t]+=grad*h/(l*u);
          GL[b*T+t]-=grad*h*q/l;
          GR[b*nr+t]+=grad*(scalar_t(1)-h);
          GS[b*ns+t]-=grad*h*q/u;
        }
        const scalar_t grad=GL[b*T+t], q=z/(u*v);
        DC[b*nc]+=grad*(q-L[b*T+t-1]);
        DY[b*T+t]+=grad*a/(u*v);
        GL[b*T+t-1]+=grad*(scalar_t(1)-a);
        if(m) GS[b*ns+t]-=grad*a*q/u;
        if(n) GR[b*nr+t]-=grad*a*q/v;
      }
      const scalar_t u=m ? S[b*ns] : scalar_t(1), v=n ? R[b*nr] : scalar_t(1);
      DY[b*T]+=GL[b*T]/(u*v);
      if(m) {
        GS[b*ns]-=GL[b*T]*L[b*T]/u;
        GS[b*ns]+=GS[b*ns+m];
        for(int64_t j=0;j<m;++j) DI[b*m+j]=GS[b*ns+j];
      }
      if(n) {
        GR[b*nr]-=GL[b*T]*L[b*T]/v;
        GR[b*nr]+=GR[b*nr+n];
        for(int64_t j=0;j<n;++j) DJ[b*n+j]=GR[b*nr+j];
      }
    }
  });
  return {dy,dc,di,dj};
}

#ifdef __APPLE__
template <typename T>
void recurrent_product(const T* left, const T* weight, T* result, int B, int H, bool backward) {
  const int N=backward ? H : 4*H, K=backward ? 4*H : H;
  const auto transpose=backward ? CblasNoTrans : CblasTrans;
  if constexpr (std::is_same_v<T,float>) {
    cblas_sgemm(CblasRowMajor,CblasNoTrans,transpose,B,N,K,1,left,K,weight,H,0,result,N);
  } else {
    cblas_dgemm(CblasRowMajor,CblasNoTrans,transpose,B,N,K,1,left,K,weight,H,0,result,N);
  }
}
#endif

// Keep platform directives outside AT_DISPATCH arguments for MSVC's preprocessor.
template <typename T>
void serial_recurrent_forward(const Tensor& out, const Tensor& zero, const Tensor& wh,
                              const Tensor& recurrent_weights, const Tensor& previous,
                              Tensor& rec, int64_t t, int64_t B, int64_t H) {
#ifdef __APPLE__
  recurrent_product(t ? out.data_ptr<T>()+(t-1)*B*H : zero.data_ptr<T>(),
                    wh.data_ptr<T>(),rec.data_ptr<T>(),B,H,false);
#else
  at::mm_out(rec,previous,recurrent_weights);
#endif
}

template <typename T>
void serial_recurrent_backward(const Tensor& dg, const Tensor& wh, Tensor& dh,
                               int64_t t, int64_t B, int64_t H) {
#ifdef __APPLE__
  recurrent_product(dg.data_ptr<T>()+t*B*4*H,wh.data_ptr<T>(),dh.data_ptr<T>(),B,H,true);
#else
  at::mm_out(dh,dg.select(0,t),wh);
#endif
}

// Single-row activation helpers share formulas between serial and parallel scans.
template <typename T>
void activate_row(T* gates, const T* recurrent, const T* bias,
                  const T* previous, T* state, T* output, int64_t H) {
  using Vec=at::vec::Vectorized<T>;
  int64_t j=0;
  for(;j+Vec::size()<=H;j+=Vec::size()) {
    const Vec one(T(1));
    auto iv=(one+(-(Vec::loadu(gates+j)+(Vec::loadu(recurrent+j)+Vec::loadu(bias+j)))).exp()).reciprocal();
    auto fv=(one+(-(Vec::loadu(gates+j+H)+(Vec::loadu(recurrent+j+H)+Vec::loadu(bias+j+H)))).exp()).reciprocal();
    auto gv=(Vec::loadu(gates+j+2*H)+(Vec::loadu(recurrent+j+2*H)+Vec::loadu(bias+j+2*H))).tanh();
    auto ov=(one+(-(Vec::loadu(gates+j+3*H)+(Vec::loadu(recurrent+j+3*H)+Vec::loadu(bias+j+3*H)))).exp()).reciprocal();
    auto cv=fv*(previous ? Vec::loadu(previous+j) : Vec(T(0)))+iv*gv;
    cv.store(state+j); (ov*cv.tanh()).store(output+j);
    iv.store(gates+j); fv.store(gates+j+H); gv.store(gates+j+2*H); ov.store(gates+j+3*H);
  }
  for(;j<H;++j) {
    const T iv=T(1)/(T(1)+std::exp(-(gates[j]+(recurrent[j]+bias[j]))));
    const T fv=T(1)/(T(1)+std::exp(-(gates[j+H]+(recurrent[j+H]+bias[j+H]))));
    const T gv=std::tanh(gates[j+2*H]+(recurrent[j+2*H]+bias[j+2*H]));
    const T ov=T(1)/(T(1)+std::exp(-(gates[j+3*H]+(recurrent[j+3*H]+bias[j+3*H]))));
    const T cv=fv*(previous ? previous[j] : T(0))+iv*gv;
    state[j]=cv; output[j]=ov*std::tanh(cv);
    gates[j]=iv; gates[j+H]=fv; gates[j+2*H]=gv; gates[j+3*H]=ov;
  }
}

template <typename T>
void gradient_row(const T* gates,const T* state,const T* previous,const T* upstream,
                  const T* hidden_grad,T* cell_grad,T* gate_grad,int64_t H) {
  using Vec=at::vec::Vectorized<T>;
  int64_t j=0;
  for(;j+Vec::size()<=H;j+=Vec::size()) {
    const Vec one(T(1));
    auto iv=Vec::loadu(gates+j), fv=Vec::loadu(gates+j+H), gv=Vec::loadu(gates+j+2*H), ov=Vec::loadu(gates+j+3*H);
    auto tanhc=Vec::loadu(state+j).tanh(), hy=Vec::loadu(upstream+j)+Vec::loadu(hidden_grad+j);
    auto cy=Vec::loadu(cell_grad+j)+hy*ov*(one-tanhc*tanhc);
    (cy*gv*iv*(one-iv)).store(gate_grad+j);
    (cy*(previous ? Vec::loadu(previous+j) : Vec(T(0)))*fv*(one-fv)).store(gate_grad+j+H);
    (cy*iv*(one-gv*gv)).store(gate_grad+j+2*H);
    (hy*tanhc*ov*(one-ov)).store(gate_grad+j+3*H);
    (cy*fv).store(cell_grad+j);
  }
  for(;j<H;++j) {
    const T iv=gates[j],fv=gates[j+H],gv=gates[j+2*H],ov=gates[j+3*H];
    const T tanhc=std::tanh(state[j]),hy=upstream[j]+hidden_grad[j];
    const T cy=cell_grad[j]+hy*ov*(T(1)-tanhc*tanhc);
    gate_grad[j]=cy*gv*iv*(T(1)-iv);
    gate_grad[j+H]=cy*(previous ? previous[j] : T(0))*fv*(T(1)-fv);
    gate_grad[j+2*H]=cy*iv*(T(1)-gv*gv);
    gate_grad[j+3*H]=hy*tanhc*ov*(T(1)-ov);
    cell_grad[j]=cy*fv;
  }
}

std::vector<Tensor> lstm_forward(Tensor x, Tensor wi, Tensor wh, Tensor bi, Tensor bh) {
  TORCH_CHECK(x.device().is_cpu() && wi.device().is_cpu() && wh.device().is_cpu() && bi.device().is_cpu() && bh.device().is_cpu(), "LSTM scan requires CPU tensors");
  TORCH_CHECK(x.dim()==3 && wi.dim()==2 && wh.dim()==2 && bi.dim()==1 && bh.dim()==1, "LSTM scan dimensionality mismatch");
  const int64_t T=x.size(0), B=x.size(1), I=x.size(2), H=wh.size(1);
  TORCH_CHECK(T>0 && wi.size(0)==4*H && wi.size(1)==I && wh.size(0)==4*H && bi.size(0)==4*H && bh.size(0)==4*H, "LSTM scan shape mismatch");
  TORCH_CHECK(x.is_contiguous() && wi.is_contiguous() && wh.is_contiguous() && bi.is_contiguous() && bh.is_contiguous(), "LSTM scan requires contiguous inputs");
  TORCH_CHECK(x.scalar_type()==wi.scalar_type() && x.scalar_type()==wh.scalar_type() && x.scalar_type()==bi.scalar_type() && x.scalar_type()==bh.scalar_type(), "LSTM scan dtype mismatch");
  auto gates=at::addmm(bi,x.reshape({T*B,I}),wi.t()).reshape({T,B,4*H});
  auto states=at::empty({T,B,H},x.options()), out=at::empty_like(states);
  auto zero=at::zeros({B,H},x.options()),rec=at::empty({B,4*H},x.options());
#if defined(__APPLE__) && defined(_OPENMP)
  if(at::get_num_threads()>1 && B>1) {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"lstm_parallel_forward",[&] {
      auto *G=gates.data_ptr<scalar_t>(), *C=states.data_ptr<scalar_t>(), *Y=out.data_ptr<scalar_t>(), *R=rec.data_ptr<scalar_t>();
      const auto *W=wh.data_ptr<scalar_t>(), *Z=zero.data_ptr<scalar_t>(), *BH=bh.data_ptr<scalar_t>();
      at::parallel_for(0,B,1,[&](int64_t begin,int64_t end) {
        // Only raw pointers and BLAS: no Tensor operations inside parallel_for.
        for(int64_t b=begin;b<end;++b) for(int64_t t=0;t<T;++t) {
          const int64_t k=t*B*H+b*H;
          recurrent_product(t ? Y+k-B*H : Z+b*H,W,R+b*4*H,1,H,false);
          activate_row(G+t*B*4*H+b*4*H,R+b*4*H,BH,t ? C+k-B*H : nullptr,C+k,Y+k,H);
        }
      });
    });
    return {out,states,gates};
  }
#endif
  auto recurrent_weights=wh.t(),prev=zero;
  for(int64_t t=0;t<T;++t) {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"lstm_scan_forward",[&] {
      auto *G=gates.data_ptr<scalar_t>(), *C=states.data_ptr<scalar_t>(), *Y=out.data_ptr<scalar_t>(), *R=rec.data_ptr<scalar_t>();
      const auto *BH=bh.data_ptr<scalar_t>();
      serial_recurrent_forward<scalar_t>(out,zero,wh,recurrent_weights,prev,rec,t,B,H);
      for(int64_t b=0;b<B;++b) {
        const int64_t k=t*B*H+b*H;
        activate_row(G+t*B*4*H+b*4*H,R+b*4*H,BH,t ? C+k-B*H : nullptr,C+k,Y+k,H);
      }
    });
#ifndef __APPLE__
    prev=out.select(0,t);
#endif
  }
  return {out,states,gates};
}

std::vector<Tensor> lstm_backward(Tensor x,Tensor wi,Tensor wh,Tensor out,Tensor states,Tensor gates,Tensor gradient) {
  const int64_t T=x.size(0), B=x.size(1), I=x.size(2), H=wh.size(1);
  auto dg=at::empty_like(gates), dh=at::zeros({B,H},x.options()), dc=at::zeros_like(dh);
  bool parallel=false;
#if defined(__APPLE__) && defined(_OPENMP)
  parallel=at::get_num_threads()>1 && B>1;
  if(parallel) {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"lstm_parallel_backward",[&] {
      const auto *G=gates.data_ptr<scalar_t>(), *C=states.data_ptr<scalar_t>(), *DY=gradient.data_ptr<scalar_t>(), *W=wh.data_ptr<scalar_t>();
      auto *DG=dg.data_ptr<scalar_t>(), *DH=dh.data_ptr<scalar_t>(), *DC=dc.data_ptr<scalar_t>();
      at::parallel_for(0,B,1,[&](int64_t begin,int64_t end) {
        for(int64_t b=begin;b<end;++b) for(int64_t t=T-1;t>=0;--t) {
          const int64_t k=t*B*H+b*H,g=t*B*4*H+b*4*H;
          gradient_row(G+g,C+k,t ? C+k-B*H : nullptr,DY+k,DH+b*H,DC+b*H,DG+g,H);
          recurrent_product(DG+g,W,DH+b*H,1,H,true);
        }
      });
    });
  }
#endif
  if(!parallel) for(int64_t t=T-1;t>=0;--t) {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"lstm_scan_backward",[&] {
      const auto *G=gates.data_ptr<scalar_t>(), *C=states.data_ptr<scalar_t>(), *DY=gradient.data_ptr<scalar_t>();
      auto *DG=dg.data_ptr<scalar_t>(), *DH=dh.data_ptr<scalar_t>(), *DC=dc.data_ptr<scalar_t>();
      for(int64_t b=0;b<B;++b) {
        const int64_t k=t*B*H+b*H,g=t*B*4*H+b*4*H;
        gradient_row(G+g,C+k,t ? C+k-B*H : nullptr,DY+k,DH+b*H,DC+b*H,DG+g,H);
      }
      serial_recurrent_backward<scalar_t>(dg,wh,dh,t,B,H);
    });
  }
  auto flat=dg.reshape({T*B,4*H});
  auto dx=at::mm(flat,wi).reshape_as(x), dwi=at::mm(flat.t(),x.reshape({T*B,I}));
  auto dwh=T>1 ? at::mm(dg.narrow(0,1,T-1).reshape({(T-1)*B,4*H}).t(),
                         out.narrow(0,0,T-1).reshape({(T-1)*B,H})) : at::zeros_like(wh);
  auto db=flat.sum(at::IntArrayRef{0});
  return {dx,dwi,dwh,db,db.clone()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("lstm_forward", &lstm_forward);
  module.def("lstm_backward", &lstm_backward);
  module.def("forward", &scan_forward);
  module.def("backward", &scan_backward);
}

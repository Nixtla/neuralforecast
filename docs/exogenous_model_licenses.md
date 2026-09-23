# Attribution for exogenous model ports

## CrossLinear

`neuralforecast/models/crosslinear.py` adapts the MS forecasting path from
https://github.com/mumiao2000/CrossLinear/blob/main/models/CrossLinear.py,
blob `594ad517e8e3f188d77ce09d995fc6f3062827aa`.
The changes add NF window/training integration, validation and a point-output
contract; the source architecture and target-last convention are retained.
The upstream license (`LICENSE.md`, blob
`f35efc6f289bebc1865baa52695d07cbbf2bb3ff`) follows in full.

The MIT License (MIT)
=====================

Copyright © 2025 LINKE Lab

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

## Timer-XL / OpenLTM

`neuralforecast/models/timerxl.py` adapts the official OpenLTM implementation:
https://github.com/thuml/OpenLTM. It fuses Q/K/V projections, uses linear layers
in place of equivalent 1x1 convolutions, removes unrelated attention variants,
and exposes the final target token's horizon to NF training. It does not claim
upstream all-token pretraining or pretrained-checkpoint compatibility.

Reviewed source blobs:

| File | Blob |
|---|---|
| `models/timer_xl.py` | `a554cf3b64ed1513953430e4ad8495b5e24d1b5a` |
| `layers/SelfAttention_Family.py` | `36e24a839d9e5febcdac59836446b21ddad9da6a` |
| `layers/Attn_Bias.py` | `4214e4ba7b0bb4864447746764d5ab3e178593ec` |
| `layers/Attn_Projection.py` | `0002744c90750981382414b417150d53bec76815` |
| `layers/Transformer_EncDec.py` | `963b7984e425aa01279e592d33e4920b07a31983` |
| `utils/masking.py` | `63a0bf5f280c36045bad70d059107931723c320d` |

The upstream `LICENSE` (blob `6f6856e7421037b9d06cf954b9e63d32aeaac695`)
follows in full. This is OpenLTM's MIT license, not an assumption based on the
separate Timer-XL repository's license.

MIT License

Copyright (c) 2022 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## External backends

Chronos-2 (Amazon), Moirai/Moirai-MoE (Salesforce), TimesFM (Google), TinyTimeMixer
(IBM), and Toto (Datadog) are invoked through their official packages. Their
implementation source and weights are not copied into this repository. Users
must comply separately with the selected package and checkpoint license. See
`exogenous_models.md` for exact upstream references and supported generations.

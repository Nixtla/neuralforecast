# taken from: https://raw.githubusercontent.com/pete88b/decision_tree/master/test_nbs.py

# We need a "test_" file with "test_" functions to make it easy to run with pytest


# couple of example "test_" functions

# import nbdev.test
# def test_run():
#     print('running nbdev.test.test_nb("20_models.ipynb") ...')
#     nbdev.test.test_nb('20_models.ipynb')

# import os
# def test_run():
#     print('running nbdev_test_nbs...')
#     os.system('nbdev_test_nbs')


# set-up a "before test" callback handler that will modify the notebook before it is run
import os
from pathlib import Path
import time
import glob
import nbformat
from nbdev.imports import get_config, parallel
from nbdev.export import read_nb,find_default_export,is_export,split_flags_and_code
from nbdev.test import get_all_flags,NoExportPreprocessor

def before_test(nb):
    "callback that will import modules and run cells that are not exported"
    default_export=find_default_export(nb['cells'])
    exports = [is_export(c, default_export) for c in nb['cells']]
    imports = ''
    # exclude exported, notebook2script calls etc
    things_to_exclude = ['notebook2script']
    cells = [(i,c,e) for i,(c,e) in enumerate(zip(nb['cells'],exports)) if c['cell_type']=='code']
    for i,c,e in cells:
        if e: 
            c['cell_type']='exclude'             # if it's exported to the library, don't run as test
            for line in split_flags_and_code(c): # but we might still need to run import statements
                if 'import' in line: imports += f'{line}\n'
            continue 
        for thing_to_exclude in things_to_exclude: # TODO: is this too coarse? maybe just exclude specific lines?
            if thing_to_exclude in c['source']: 
                c['cell_type']='exclude'
                continue
    
    nb['cells'].insert(0,nbformat.v4.new_code_cell(imports))
    
    # import everything from modules written to by this notebook
    for export in {export[0] for export in exports if export}:
        export_parts=export.split('.')
        b=export_parts.pop()
        export_parts.insert(0, get_config().lib_name)
        a='.'.join(export_parts)
        src=f"""
from {a} import {b}
for o in dir({b}):
    exec(f'from {a}.{b} import {{o}}')"""
        nb['cells'].insert(0,nbformat.v4.new_code_cell(src))
    return nb

# uncomment to see current nbdev behaviour
# i.e. use a before test callback that does nothing
# def before_test(nb): return nb

# If nbdev.test.test_nb knew to call our "before test" callback, the rest of this script could be just the following 3 lines
# def test_run():
#     from nbdev.cli import nbdev_test_nbs
#     nbdev_test_nbs.__wrapped__()

# until it does ... we need to duplicate a few chunks of nbdev
def _test_nb(fn, flags=None):
    "Execute tests in notebook in `fn` with `flags`"
    os.environ["IN_TEST"] = '1'
    if flags is None: flags = []
    try:
        nb = read_nb(fn)
        nb = before_test(nb) # <- THIS is the only change to nbdev code
        for f in get_all_flags(nb['cells']):
            if f not in flags: return
        ep = NoExportPreprocessor(flags, timeout=600, kernel_name='python3')
        pnb = nbformat.from_dict(nb)
        ep.preprocess(pnb)
    finally: os.environ.pop("IN_TEST")

def _test_one(fname, flags=None, verbose=True):
    print(f"testing: {fname}")
    start = time.time()
    try: 
        _test_nb(fname, flags=flags)
        return True,time.time()-start
    except Exception as e: 
        if "Kernel died before replying to kernel_info" in str(e):
            time.sleep(random.random())
            _test_one(fname, flags=flags)
        if verbose: print(f'Error in {fname}:\n{e}')
        return False,time.time()-start

def nbdev_test_nbs(fname=None,flags=None,n_workers=None,verbose=True,timing=False):
    """
    fname:Param("A notebook name or glob to convert", str)=None,
    flags:Param("Space separated list of flags", str)=None,
    n_workers:Param("Number of workers to use", int)=None,
    verbose:Param("Print errors along the way", bool)=True,
    timing:Param("Timing each notebook to see the ones are slow", bool)=False
    """
    "Test in parallel the notebooks matching `fname`, passing along `flags`"
    if flags is not None: flags = flags.split(' ')
    if fname is None: 
        files = [f for f in Path(get_config().nbs_path).glob('*.ipynb') if not f.name.startswith('_')]
    else: files = glob.glob(fname)
    files = [Path(f).absolute() for f in sorted(files)]
    if len(files)==1 and n_workers is None: n_workers=0
    # make sure we are inside the notebook folder of the project
    os.chdir(get_config().nbs_path)
    results = parallel(_test_one, files, flags=flags, verbose=verbose, n_workers=n_workers)
    passed,times = [r[0] for r in results],[r[1] for r in results]
    if all(passed): print("All tests are passing!")
    else:
        msg = "The following notebooks failed:\n"
        raise Exception(msg + '\n'.join([f.name for p,f in zip(passed,files) if not p]))
    if timing:
        for i,t in sorted(enumerate(times), key=lambda o:o[1], reverse=True): 
            print(f"Notebook {files[i].name} took {int(t)} seconds")

def test_run():
    # now we can "nbdev_test_nbs" and have our "before test" callback called
    # nbdev_test_nbs('00_core.ipynb') # Use this line to test a single notebook
    nbdev_test_nbs(flags='distributed', n_workers=1)

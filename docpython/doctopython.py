
# coding: utf-8

import os
import shutil
from git import Repo
from pathlib import Path
import subprocess
import jinja2
import stat

remote_repo = "https://github.com/GeoscienceAustralia/agdc_statistics.git"
modules_path = '/g/data/u46/users/hr8696'
module_name = 'agdc-statistics'
module_version = 'dev'
install_root = os.path.join(modules_path, module_name, module_version)
temp_directory = os.environ["TMPDIR"]
checkout_path = os.path.join(temp_directory, module_name)
python_version = "3.6"
python_path = os.path.join(install_root, 'lib', 'python' + python_version, 'site-packages')
os.environ['PYTHONPATH'] = python_path
module_dest = os.path.join(modules_path, 'modulefiles', module_name)
module_dest_file = os.path.join(module_dest, module_version)
template = str(Path(__file__).parents[0].absolute())


src_name = 'module_template.j2'

template_context = {
    'remote_repo': remote_repo,
    'modules_path': modules_path,
    'module_name': module_name,
    'module_version': module_version,
    'install_root': install_root,
    'checkout_path': checkout_path,
    'python_path': python_path,
    }


def deploypackage():
    os.chmod(install_root, 0o700)
    try:
        shutil.rmtree(checkout_path)
    except FileNotFoundError:
        pass

    os.makedirs(checkout_path)

    # change directory to temp directory
    os.chdir(checkout_path)

    Repo.clone_from(remote_repo, checkout_path, branch='master')

    if not os.path.isdir(install_root):
        os.makedirs(install_root)

    if not os.path.isdir(pyhton_path):
        os.makedirs(python_path)
    
    package = ("python setup.py clean && python setup.py install --prefix " + install_root)
    subprocess.run(package, shell=True)
    return 'success'


def load_template(name):
    if name == 'index.html':
        return '...'


def run(template, template_context, module_dest, module_dest_file):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template))
    if not os.path.isdir(module_dest):
        os.makedirs(module_dest)
    tmpl = env.get_template(src_name)
    with open(module_dest_file, 'w') as fd:
        fd.write(tmpl.render(**template_context))
    os.chmod(module_dest_file, 0o660)
    return True


if __name__ == '__main__':
    deploypackage()
    run(template, template_context, module_dest, module_dest_file)
    os.chmod(install_root, 0o755)

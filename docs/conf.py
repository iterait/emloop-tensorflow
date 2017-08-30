import sys
import pkg_resources

sys.path.insert(0, "_base")
from conf import *

autoapi_modules = {
    'cxflow_tensorflow': {
        'prune': True
    }
}

# General information about the project.                         
project = 'cxflow-tensorflow' 
copyright = '2017, Cognexa Solutions s.r.o.'                     
author = 'Blazek Adam, Belohlavek Petr, Matzner Filip'

# The short X.Y version.
version = '.'.join(pkg_resources.get_distribution("cxflow-tensorflow").version.split('.')[:2])
# The full version, including alpha/beta/rc tags.
release = pkg_resources.get_distribution("cxflow-tensorflow").version

html_static_path.append("_static")

html_theme_options.update({
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': "cxflow-tensorflow",

    # Tab name for entire site. (Default: "Site")
    'navbar_site_name': "Pages",

    # A list of tuples containing pages or urls to link to.
    'navbar_links': [
        ("Getting Started", "getting_started"),
        ("Tutorial", "tutorial"),
        ("Multi GPU models", "multi_gpu"),
        ("API Reference", "cxflow_tensorflow/index"),
    ],
})


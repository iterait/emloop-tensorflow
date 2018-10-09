import sys
import pkg_resources

sys.path.insert(0, "_base")
from conf import *

autoapi_modules = {
    'emloop_tensorflow': {
        'prune': True
    }
}

# General information about the project.                         
project = 'emloop-tensorflow' 
copyright = '2018, Iterait a.s.'                     
author = 'Blazek Adam, Belohlavek Petr, Matzner Filip'

# The short X.Y version.
version = '.'.join(pkg_resources.get_distribution("emloop-tensorflow").version.split('.')[:2])
# The full version, including alpha/beta/rc tags.
release = pkg_resources.get_distribution("emloop-tensorflow").version

html_static_path.append("_static")

html_context.update(analytics_id="UA-108931454-1")

html_theme_options.update({
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': "emloop-tensorflow",

    # Tab name for entire site. (Default: "Site")
    'navbar_site_name': "Pages",

    # A list of tuples containing pages or urls to link to.
    'navbar_links': [
        ("Getting Started", "getting_started"),
        ("Tutorial", "tutorial"),
        ("Model Regularization", "regularization"),
        ("Multi GPU models", "multi_gpu"),
        ("API Reference", "emloop_tensorflow/index"),
    ],
})


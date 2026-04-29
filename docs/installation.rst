Getting FASTAR
==================================

.. code-block:: python

   python3 -m pip install fastar-astro

This will install it in the current environment, but it is recommended to use
a virtual environment for the installation (using ``venv``, ``conda``, ``uv`` or the
tool of your choosing).

.. note::
   The name of the module is ``fastar``, meaning that in your code you must use
   ``import fastar``. However, due to a name clash on PyPi, **only** for downloading
   the package you need to use ``fastar-astro``.

Development installation
~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to modify the ``fastar`` code, we recommend to use the provided
``Makefile`` to setup your environment. It uses `uv <https://docs.astral.sh/uv/>`__
to setup a virtual environment and it should be transparent to the developer.

.. code-block:: bash

   git clone https://github.com/inavarro/fastar.git
   cd fastar/
   make install-dev
   # ... do your development and test them
   make tests
   # ... add your changes and do a commit
   # git add ...
   # git commit ...
   # This will trigger some automatic checks (format, linting,...) that you must
   # address before committing (some of them are fixed automatically).
   git push

Changes are welcome! But remember that it is always safer to work on your local
fork and only when your development is finished you can proceed to do a PR.

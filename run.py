#!/usr/bin/env python
from app import app
app.run(debug = True)
app._static_folder = '/app/static/'

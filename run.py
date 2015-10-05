#!/usr/bin/env python
from app import app
app.run(port=5000, debug = True)   #host='0,0,0,0', 
app._static_folder = '/app/static/'

"""Sassy ROSMASTER, a WendyOS rover app."""
def main():
    import os
    import uvicorn
    host = os.getenv('HOST', '127.0.0.1' if os.getenv('ROVER_MODE') == 'hardware' else '0.0.0.0')
    uvicorn.run('sassy_rosmaster.server:app', host=host, port=int(os.getenv('PORT', '8096')))

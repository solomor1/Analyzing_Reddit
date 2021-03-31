from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple
from server import server
from dash_test  import app as app1
from app import app as app2

app = DispatcherMiddleware(server,
                           {'/myflaskapp': app2})
if __name__ == '__main__':
    run_simple('localhost', 5000, app, use_reloader=True)

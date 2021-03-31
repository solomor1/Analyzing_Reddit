from flask import Flask, render_template

server = Flask(__name__)

@server.route('/')
def server_root():
  return render_template('graph.html')

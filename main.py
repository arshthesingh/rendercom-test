from app import create_app
from flask import Flask, jsonify, send_from_directory

app = create_app()



# Route to serve the Swagger YAML file
@app.route("/static/swagger.yaml")
def serve_swagger_yaml():
    return send_from_directory("static", "swagger.yaml")




if __name__ == "__main__":
    app.run(debug=True)

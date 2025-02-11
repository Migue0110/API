from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS

from modelo.analisis_depresivo import Analizador

app = Flask(__name__)
CORS(app)
analizar = Analizador()

api = Api(
    app, 
    version='1.0', 
    title='Analizador de Textos',
    description='Analizador de Textos')

ns = api.namespace('Analizador')

analizador = api.parser()

analizador.add_argument(
    'Comentario',
    type=str, 
    required=True,
    help='Ingrese un texto para analizar ...', 
    location='args')

resource_fields = api.model('Resource', {
    'Sentimiento': fields.String,
})

@ns.route('/Sentimientos')
class AnalizadorApi(Resource):

    @api.doc(parser=analizador)
    @api.marshal_with(resource_fields)
    def get(self):
        args = analizador.parse_args()

        return{
            'Sentimiento': analizar.analizar(args['Comentario'])
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
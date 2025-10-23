# SHIELDS-SMS
PROYECTO FINAL DE INGENIERIA DE SOFTWARE

# Ejecución del MVP
## requisitos:
- pickle
- uvicorn
- fastapi
- pydantic
- pandas
- numpy
- scikit-learn
- nltk
- scipy

## train_model:
Es necesario ejecutar este archivo una vez al menos para entrenar y guardar el modelo con mejor rendimiento

## Iniciar la api
Es necesario ejecutar los siguiente comandos en la terminal:
`cd src/api`
`uvicorn api:app --reload`

## Acceso a la api
En la dirección http://127.0.0.1:8000/docs
Damos click en "try it out", cambiamos el contenido a un sms simulado y damos a execute

# set base image (host OS)
FROM python:3.6

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/iris-app

# copy the dependencies file to the working directory
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/models/app.py ./src/models/

# copy model object
COPY models/final_model.pkl ./models/

# command to run on container start
# CMD ["echo", "Hello World"]
CMD [ "python", "./src/models/app.py" ]

# # set app port
# EXPOSE 8080

# ENTRYPOINT [ "python" ] 

# # Run app.py when the container launches
# CMD [ "./src/models/app.py", "run", "--host", "0.0.0.0" ] 
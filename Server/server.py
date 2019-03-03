#!/usr/bin/python3

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from urllib.parse import parse_qs
from threading import Thread, Lock
from queue import Queue, Empty
import json
import os
import sys, getopt
import csv
from datetime import datetime

# Global Variables
g_model_server = None

class ModelServer:
    format_training = {"x" : "float",
                       "y" : "float",
                       "z" : "float",
                       "t" : "int",
                       "key" : "str",
                       "sensor-type" : "str" }

    def isParameterValid(self, data, data_format):
        for key in data_format.keys():
            if (data.get(key) is None):
                return False
        return True

    def threadProcessTrainingData(self, queue, lock, client_address):
        while True:
            try:
                data = queue.get(timeout = self.timeout)

                lock.acquire()
                if self.isParameterValid(data, format_training):
                    sensor_type = data["sensor-type"]
                    if (sensor_type == "accelerometer"):
                        self.training_output_accelerometer[client_address]["csv_file"].writerow([data["x"],
                                                                                                 data["y"],
                                                                                                 data["z"],
                                                                                                 data["t"],
                                                                                                 data["key"]])
                    elif (sensor_type == "gyroscope"):
                        self.training_output_gyroscope[client_address]["csv_file"].writerow([data["x"],
                                                                                             data["y"],
                                                                                             data["z"],
                                                                                             data["t"],
                                                                                             data["key"]])
                    elif (sensor_type == "NULL"):
                        lock.release()
                        break
                    else:
                        pass
                else:
                    pass
                lock.release()
            except Empty:
                break

        # Clean up thread
        self.training_output_accelerometer[client_address]["fh_file"].close()
        self.training_output_gyroscope[client_address]["fh_file"].close()
        del self.threads_training[queue]
        del self.registered_training[client_address]
        del self.training_output_accelerometer[client_address]
        del self.training_output_gyroscope[client_address]
        print (" - Client disconnected: {}\n".format(client_address))
        return

    def threadProcessInferenceData(self, queue, lock, client_address):
        while True:
            try:
                data = queue.get(timeout = self.timeout)

                lock.acquire()
                # Infer data
                lock.release()
            except Empty:
                del self.threads_inference[queue]
                del self.registered_inference[client_address]
                break
        return

    def __init__(self, timeout):
        self.threads_training = {}
        self.threads_inference = {}
        self.registered_training = {}
        self.registered_inference = {}
        self.training_output_accelerometer = {}
        self.training_output_gyroscope = {}
        self.timeout = timeout

        if (not os.path.isdir("Model")):
            os.mkdir("Model")

    def registerListenerTraining(self, client_address):
        # Set up output file
        current_time = datetime.now()
        path_output_file = "{}-{}-{}_{}:{}:{}".format(current_time.year,
                                                      current_time.month,
                                                      current_time.day,
                                                      current_time.hour,
                                                      current_time.minute,
                                                      current_time.second)
        fh_output_file_accelerometer = open(os.path.join("Model", "{}_accelerometer.csv".format(path_output_file)), "w+")
        csv_output_file_accelerometer = csv.writer(fh_output_file_accelerometer, delimiter = ",")
        self.training_output_accelerometer[client_address] = {"fh_file" : fh_output_file_accelerometer,
                                                              "csv_file" : csv_output_file_accelerometer}
        csv_output_file_accelerometer.writerow(["x", "y", "z", "t", "key"])
        fh_output_file_gyroscope = open(os.path.join("Model", "{}_gyroscope.csv".format(path_output_file)), "w+")
        csv_output_file_gyroscope = csv.writer(fh_output_file_gyroscope, delimiter = ",")
        self.training_output_gyroscope[client_address] = {"fh_file" : fh_output_file_gyroscope,
                                                          "csv_file" : csv_output_file_gyroscope}
        csv_output_file_gyroscope.writerow(["x", "y", "z", "t", "key"])

        queue = Queue()
        lock = Lock()
        self.threads_training[queue] = Thread(target = threadProcessTrainingData,
                                              kwargs = {"queue" : queue,
                                                        "lock" : lock,
                                                        "client_address" : client_address})
        self.threads_training[queue].setDaemon(True)
        self.threads_training[queue].start()
        self.registered_training[client_address] = queue
        return

    def registerListenerInference(self, client_address):
        queue = Queue()
        lock = Lock()
        self.threads_inference[queue] = Thread(target = threadProcessInferenceData,
                                               kwargs = {"queue": queue,
                                                         "lock" : log,
                                                         "client_address" : client_address})
        self.threads_inference[queue].setDaemon(True)
        self.threads_inference[queue].start()
        self.registered_inference[client_address] = queue
        return

    def isClientRegisteredTraining(self, client_address):
        if (self.registered_training.get(client_address) is None):
            return None
        else:
            return self.registered_training[client_address]

    def isClientRegisteredInference(self, client_address):
        if (self.registered_inference.get(client_address) is None):
            return False
        else:
            return True

    def getQueueTraining(self, client_address):
        if (self.registered_training.get(client_address) is None):
            return None
        else:
            return(self.registered_training[client_address])

    def getQueueInference(self, client_address):
        if (self.registered_inference.get(client_address) is None):
            return None
        else:
            return(self.registered_inference[client_address])

class HTTPRequestHandler(BaseHTTPRequestHandler):
    def respond(self, status, content_type, content):
        self.send_response(status)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content)

    def sendNACK(self):
        response = {"status-code" : -1}
        self.respond(200, "text/html", bytes(json.dumps(response), "UTF-8"))

    def sendACK(self):
        response = {"status-code" : 0}
        self.respond(200, "text/html", bytes(json.dumps(response), "UTF-8"))

    def splitPath(self, path):
        path_split = path.split("/")
        for index in range(len(path_split)):
            if path_split[index] == '':
                del path_split[index]
        return(path_split)

    def do_POST(self):
        try:
            global g_model_server

            parsed_path = urlparse(self.path)
            path_split = self.splitPath(parsed_path.path)
            parameters = parse_qs(self.rfile.read(int(self.headers['Content-Length'])).decode("utf-8"))
            if path_split[0] == "api":
                client_address = self.client_address
                if path_split[1] == "post-training":
                    if (not g_model_server.isClientRegisteredTraining(client_address)):
                        print(" + Created new client: {}\n".format(client_address))
                        g_model_server.registerListenerTraining(client_address)

                    queue = g_model_server.getQueueTraining(client_address)
                    if (queue is not None):
                        if (parameters.get("data") is None):
                            print(" ! Received invalid POST data format: {}\n".format(client_address))
                            self.sendNACK()
                        else:
                            queue.put(json.loads(parameters["data"]))
                            self.sendACK()
                    else:
                        print(" ! Queue not found: {}\n".format(client_address))
                        self.sendNACK()
                elif path_split[1] == "post-inferrence":
                    if (not g_model_server.isClientRegisteredInference(client_address)):
                        print(" + Created new client: {}\n".format(client_address))
                        g_model_server.registerListenerInference(client_address)

                    queue = g_model_server.getQueueInference(client_address)
                    if (queue is not None):
                        if (parameters.get("data") is None):
                            print(" ! Received invalid POST data format: {}\n".format(client_address))
                            self.sendNACK()
                        else:
                            queue.put(json.loads(parameters["data"]))
                            self.sendACK()
                    else:
                        print(" ! Queue not found: {}\n".format(client_address))
                        self.sendNACK()
                else:
                    self.sendNACK()
            else:
                self.sendNACK()
        except Exception as e:
            print("Exception: {}\n".format(e))
            self.sendNACK()

def main(argv):
    # Declare necessary variables
    config_file = ''
    port = 3000
    address = "127.0.0.1"
    timeout = 300

    # Parse Arguments
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print("server.py -c <config_file>\n")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("server.py -c <config_file>\n")
            sys.exit()
        elif opt in ("-c", "--config"):
            config_file = arg

    # Make sure necessary variables are not empty
    if (config_file == ''):
        print("\n\t ! Error: Config file must not be empty\n")
        sys.exit()

    ## Import model
    fh_config_file = open(config_file, "r+")
    json_config_file = json.load(fh_config_file)
    fh_config_file.close()
    if (json_config_file.get("port") is not None):
        port = json_config_file["port"]

    if (json_config_file.get("address") is not None):
        address = json_config_file["address"]

    if (json_config_file.get("timeout") is not None):
        timeout = json_config_file["timeout"]

    # Initiate Model Server
    global g_model_server
    g_model_server = ModelServer(timeout)

    # Initiate HTTP Server
    server_address = (address, port)
    httpd = HTTPServer(server_address, HTTPRequestHandler)
    print("\n\t + Serving on {} at port {}\n".format(address, port))
    httpd.serve_forever()

if __name__ == "__main__":
    main(sys.argv[1:])

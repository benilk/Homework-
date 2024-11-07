import socket
import threading
import time

def receive_messages(client_socket):
    while True:
        try:
            message = client_socket.recv(1024).decode('utf-8')
            if message:
                print("Server:", message)
            else:
                break
        except:
            break

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            client_socket.connect(('127.0.0.1', 3000))
            print("Connected to server")
            break
        except:
            time.sleep(5)
    thread = threading.Thread(target=receive_messages, args=(client_socket,))
    thread.start()
    while True:
        message = input("You: ")
        client_socket.send(message.encode('utf-8'))

start_client()

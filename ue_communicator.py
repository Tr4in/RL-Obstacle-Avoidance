import win32pipe, win32file
import struct

class UECommunicator:
    def __init__(self, pipe):
        self.pipe = pipe

    def next_filename(self):
        print('Read next File-Name')
        filename_bytes = self.__read_bytes(1024)
        filename = filename_bytes.decode("utf-8")
        filename = filename.rstrip('\x00')
        return filename

    def get_data(self, num_laser):
        print('Get Data')
        send_data = str.encode(f"{0}")
        self.__send(send_data, self.__message_to_byte_array(0), self.__message_to_byte_array(0))
        return self.collision_happend(), self.get_speed(), self.get_steering(), self.get_laser_distances(num_laser)  

    def reset_environment(self):
        print('Reset Environment')
        reset_environment = str.encode(f"{1}")
        self.__send(reset_environment, self.__message_to_byte_array(50), self.__message_to_byte_array(0))

    def execute_action_on_unreal_agent(self, action):
        print('Execute action in Unreal Engine')
        execute_statement = str.encode(f"{2}")
        self.__send(execute_statement, self.__message_to_byte_array(30), self.__message_to_byte_array(action))

    def wait_for_environment_loading(self):
        print('Wait for Environment to load')
        environment_loading = str.encode(f"{3}")
        self.__send(environment_loading, self.__message_to_byte_array(50), self.__message_to_byte_array(0))       

    def pause_agent(self):
        print('Pause')
        pause = str.encode(f"{5}")
        self.__send(pause, self.__message_to_byte_array(50), self.__message_to_byte_array(0))       

    def request_next_filename(self):
        print('Request next Image-Filename')
        request_filename = str.encode(f"{4}")
        self.__send(request_filename, self.__message_to_byte_array(0), self.__message_to_byte_array(0))
        return self.next_filename()

    def get_speed(self):
        print('Get Speed')
        return struct.unpack('f', self.__read_bytes(4))[0]

    def get_steering(self):
        print('Get Steering')
        return struct.unpack('f', self.__read_bytes(4))[0]

    def get_laser_distance(self):
        print('Get Laser Distance')
        return struct.unpack('f', self.__read_bytes(4))[0]

    def get_laser_distances(self, num_laser):
        print('Get Laser Distances')
        bytes = self.__read_bytes(4 * num_laser)
        distances = struct.unpack('f' * num_laser, bytes)
        return list(distances)

    def collision_happend(self):
        print('Get Collision Happend')
        collision_status_bytes = self.__read_bytes(1)
        collision_status = int.from_bytes(collision_status_bytes, "little")
        return True if collision_status == 1 else False

    def __send(self, instruction, max_ticks, action):
        win32file.WriteFile(self.pipe, instruction)
        win32file.FlushFileBuffers(self.pipe)

        win32file.WriteFile(self.pipe, max_ticks)
        win32file.FlushFileBuffers(self.pipe)

        win32file.WriteFile(self.pipe, action)
        win32file.FlushFileBuffers(self.pipe)

    def __read_bytes(self, buf_size = 512):
        _, data = win32file.ReadFile(self.pipe, buf_size)
        return data

    def __message_to_byte_array(self, action):
        action_response = bytearray()
        action_response.append(action)
        return action_response


    
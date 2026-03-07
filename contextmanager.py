import logging

logging.basicConfig(level=logging.INFO)

class track_entry_and_exit():
    var1 = "Variable"
    def __init__(self, name):
        self.name = name
        logging.info('Init: %s', self.name)

    def __enter__(self):
        logging.info('Entering: %s', self.name)

    def __exit__(self, exc_type, exc, exc_tb):
        logging.info('Exiting: %s', self.name)

def load_widget():
	print("Loading widget")

with track_entry_and_exit('widget loader'):
    print('Some time consuming activity goes here')
    load_widget()

# @track_entry_and_exit('widget loader')
# def activity():
#     print('Some time consuming activity goes here')
#     load_widget()

# activity()
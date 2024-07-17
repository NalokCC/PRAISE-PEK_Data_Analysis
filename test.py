import logging
logger = logging.getLogger('testing')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(name)s: %(message)s')
file_handler = logging.FileHandler('logs/recentlog.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def main():
    logger.info('Started')
    print('hwllo world')
    logger.info('Finished')

if __name__ == '__main__':
    main()
import logging
import datetime
logger = logging.getLogger(__name__)


def logger_setup(filename):
    new_filename_parts = filename.split('data/')
    new_filename = new_filename_parts[0] + 'logging_utilities_log-' + new_filename_parts[1]
    logging.basicConfig(filename= new_filename, level=logging.INFO)
    logger.info('Started')


def log_time(beginning_flag=True, routine_name=''):
    if beginning_flag:
        logger.info('Routine {} started at {}'.format(routine_name, str(datetime.datetime.now())))
    else:
        logger.info('Routine {} finished at {}'.format(routine_name, str(datetime.datetime.now())))
        

def log_data_for_task(routine_name='', questionnaire_element='', response=''):
    logger.info('%s: Routine %s questionnaire_element %s with response %s' % (str(datetime.datetime.now()), routine_name, questionnaire_element, response))
    

def log_status_change(routine_name='', status_change_msg=''):
    logger.info('%s: Routine %s status change %s' % (str(datetime.datetime.now()), routine_name, status_change_msg))
    
    
def log_info(routine_name='', status_change_msg=''):
    logger.info('%s: Routine %s: %s' % (str(datetime.datetime.now()), routine_name, status_change_msg))
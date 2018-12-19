import os

'''Has all the basic directories for use '''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(ROOT_DIR, 'Data/Files')
MFCC_DIR = os.path.join(FILES_DIR, 'Mfcc_Files')
AUDIO_DIR = os.path.join(FILES_DIR, 'Audio_Files')
SAMPLE_DIR = os.path.join(FILES_DIR, 'Audio_Files_Samples')
SYNAPSE_DIR = os.path.join(ROOT_DIR, 'Synapse_File_Processing')

DISSERTATION_DIR = os.path.join(ROOT_DIR, 'Dissertation')
DISS_FIG_DIR = os.path.join(DISSERTATION_DIR,'Figures')
DISS_CODE_FIG_DIR = os.path.join(DISS_FIG_DIR,'code_figures')
DISS_CODE_DIR = os.path.join(DISSERTATION_DIR,'Code')
MODEL_RUN_DIR = os.path.join(DISS_CODE_DIR, 'Model_and_Run')

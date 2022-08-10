from pathlib import Path
from gui import GUI
from tensorflow import keras
from model import main
import sys


if __name__ == '__main__': 
    path = Path('./model/model.keras').absolute()

    try:
        if sys.argv[1] == 'learn':
            main()
    except IndexError:
        pass

    if path.is_file():
        model = keras.models.load_model(path)
        GUI.model = model
        GUI.web()
    else:
        main()
        model = keras.models.load_model(path)
        GUI.model = model
        GUI.web()


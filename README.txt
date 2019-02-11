* Environment
    This program needs to run under linux/MacOS,
    and use Python3.

* Dependencies
    matplotlib, numpy

* Basic usage
    To run the program doing 10-fold cross-validation on a data file in
    .txt format, simply put the text file under the root directory of
    this program, and feed the relative path to the file as the only
    argument to the program. For example, to run with file from path
    ~/decision_tree/wifi-db/test_dataset.txt, start the program with
    the following command:

    @~/decision_tree> python3 dectree.py ./wifi_db/test_dataset.txt

    Note the input file needs to be in the same format as the example
    data file clean_dataset.txt/noisy_dataset.txt. The program loads
    the text file into a numpy array and further computation is based
    on this structure.


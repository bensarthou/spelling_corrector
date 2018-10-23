import pickle

def load_db(path, suffix='10.plk'):

    f_train = open(path+'train'+suffix, 'rb')
    db_train = pickle.load(f_train)

    f_test = open(path+'test'+suffix, 'rb')
    db_test = pickle.load(f_test)

    return db_train, db_test

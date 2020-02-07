from Session import Session

def create_session(path, sql_pswd):
    session = Session(path, sql_pswd)
    session.run_session_manualy()
    # session.run_session_with_schedualer()

def run_engine_locally():
    path = '/home/victoria/original_project/'
    sql_pswd = 'vg220790'
    create_session(path, sql_pswd)

def run_engine_on_server():
    path = '/home/ubuntu/kidsalbums'
    sql_pswd = 'r1v2a3'
    create_session(path,sql_pswd)

def main():
    run_engine_locally()
    #run_engine_on_server()

if __name__ == '__main__':
    main()

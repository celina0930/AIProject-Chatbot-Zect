from app import app

if __name__ == '__main__':
    # app.debug = True
    # app.run()
    app.run(host='168.131.155.77', port=5001, debug=False)
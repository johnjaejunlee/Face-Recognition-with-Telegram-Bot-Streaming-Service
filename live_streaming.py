# live_streaming.py

from flask import Flask, render_template, Response
import face_recog
import person_db
import argparse
from Telegram_Bot import VisitorAlarmTelegramBot as telegram
from multiprocessing import Process


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(fr):
    while True:
        jpg_bytes = fr.get_jpg_bytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')


def tele():
    pdb = person_db.PersonDB()
    pdb.load_db()
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True,
                    help="Telegram Bot Token")
    ap.add_argument("--srcfile", type=int, default=0,
                    help="Video file to process. If not specified, web cam is used.")
    ap.add_argument("--threshold", default=0.5, type=float,
                    help="threshold of the similarity")
    ap.add_argument("--sbf", default=0.5, type=float,
                    help="second between frame processed (default=0.5)")
    ap.add_argument("--resize-ratio", default=1.0, type=float,
                    help="resize the frame to process (less time, less accuracy)")
    ap.add_argument("--appearance-interval", default=10, type=int,
                    help="alarm interval second between appearance (default=10)")
    args = ap.parse_args()
    fr = face_recog.FaceRecog(pdb, args)
    #fc = fr.Facerecognizer(pdb, args)
    vatb = telegram(fr, pdb, args)
    fr.register_observer(vatb)
    vatb.start_polling()
    vatb.idle()
    fr.stop_running()

@app.route('/video_feed')
def video_feed():
    pdb = person_db.PersonDB()
    pdb.load_db()
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True,
                    help="Telegram Bot Token")
    ap.add_argument("--srcfile", type=int, default=0,
                    help="Video file to process. If not specified, web cam is used.")
    ap.add_argument("--threshold", default=0.5, type=float,
                    help="threshold of the similarity")
    ap.add_argument("--sbf", default=0.5, type=float,
                    help="second between frame processed (default=0.5)")
    ap.add_argument("--resize-ratio", default=1.0, type=float,
                    help="resize the frame to process (less time, less accuracy)")
    ap.add_argument("--appearance-interval", default=10, type=int,
                    help="alarm interval second between appearance (default=10)")
    args = ap.parse_args()
    return Response(gen(face_recog.FaceRecog(pdb, args)),mimetype='multipart/x-mixed-replace; boundary=frame')

p1 = Process(target = video_feed)
p2 = Process(target = tele)
if __name__ == '__main__':

    p1.start()
    p2.start()
    app.run(host='0.0.0.0', debug=True)


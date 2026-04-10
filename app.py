import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'weatherwise_hackathon_gizli_anahtar'

# Veritabanı Konfigürasyonu (Aynı klasörde users.db adında bir dosya oluşturacak)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Veritabanı Modeli (Tablo Yapısı)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    # Şifre hash'lenerek tutulacağı için karakter sınırını uzun tutuyoruz
    password = db.Column(db.String(200), nullable=False) 

# Uygulama ilk çalıştığında veritabanı tablolarını otomatik oluştur
with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Veritabanında kullanıcıyı ara
        user = User.query.filter_by(email=email).first()
        
        # Kullanıcı var mı ve girilen şifre veritabanındaki hash ile eşleşiyor mu?
        if user and check_password_hash(user.password, password):
            return redirect(url_for('weather_dashboard'))
        else:
            flash('E-posta veya şifre hatalı! Lütfen tekrar dene.', 'error')
            return redirect(url_for('login'))
            
    # --- DİNAMİK TEMA MANTIĞI ---
    # 1. Saat bilgisini al (18:00'den sonrası gece sayılır)
    current_hour = datetime.datetime.now().hour
    is_night = current_hour >= 18 or current_hour < 6
    
    # 2. Hava durumunu belirle (Gelecekte model datasetinden gelecek)
    # Test etmek için burayı manuel değiştirebilirsin: 'sunny', 'stormy', 'rainy'
    current_weather = 'stormy' 
            
    return render_template('index.html', is_night=is_night, weather=current_weather)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        # 1. Şifreler eşleşiyor mu kontrolü
        if password != password_confirm:
            flash('Şifreler birbiriyle eşleşmiyor!', 'error')
            return redirect(url_for('register'))
            
        # 2. E-posta daha önce kullanılmış mı kontrolü
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Bu e-posta adresi zaten sistemde kayıtlı!', 'error')
            return redirect(url_for('register'))
            
        # 3. Her şey tamamsa şifreyi hash'le ve veritabanına kaydet
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(email=email, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit() # Değişiklikleri kaydet
        
        flash('Hesabın başarıyla oluşturuldu! Şimdi giriş yapabilirsin.', 'success')
        return redirect(url_for('login'))
        
    # --- DİNAMİK TEMA MANTIĞI ---
    current_hour = datetime.datetime.now().hour
    is_night = current_hour >= 18 or current_hour < 6
    current_weather = 'stormy' 
            
    return render_template('register.html', is_night=is_night, weather=current_weather)

@app.route('/dashboard')
def weather_dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
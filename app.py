# Import required libraries and modules
from flask import Flask, render_template, request
import nltk
nltk.download('stopwords')
import model

# Initialize Flask app
app = Flask(__name__)

# Predefined list of valid user IDs
valid_userid = [
    '00sab00', '1234', 'zippy', 'zburt5', 'joshua', 'dorothy w',
    'rebecca', 'walker557', 'samantha', 'raeanne', 'kimmie',
    'cassie', 'moore222'
]

# Route for the home page
@app.route('/')
def view():
    return render_template('index.html')

# Route to handle product recommendations
@app.route('/recommend', methods=['POST'])
def recommend_top5():
    user_name = request.form['User Name']

    # If user is valid, generate and display top 5 recommendations
    if user_name in valid_userid and request.method == 'POST':
        top20_products = model.recommend_products(user_name)
        get_top5 = model.top5_products(top20_products)
        return render_template(
            'index.html',
            column_names=get_top5.columns.values,
            row_data=list(get_top5.values.tolist()),
            zip=zip,
            text='Recommended products'
        )
    # Handle invalid user input
    elif user_name not in valid_userid:
        return render_template('index.html', text='No Recommendation found for the user')
    else:
        return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.debug = False
    app.run()

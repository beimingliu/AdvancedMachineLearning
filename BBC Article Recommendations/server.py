# Launch with
#
# gunicorn -D --threads 4 -b 0.0.0.0:5000 --access-logfile server.log --timeout 60 server:app glove.6B.300d.txt bbc

from flask import Flask, render_template
from doc2vec import *
import sys

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""
    the_titles = [[a[0], a[1]] for a in articles]
    return render_template('articles.html', titles = the_titles)

@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    article = [ a for a in articles if a[0]==topic+'/'+filename]
    title = article[0][1]
    text = article[0][2].split('\n\n')
    similar_articles = recommended(article[0], articles, 5)
    return render_template('article.html', title=title, body=text, similar_articles=similar_articles)

# initialization
i = sys.argv.index('server:app')
glove_filename = sys.argv[i+1]
articles_dirname = sys.argv[i+2]

gloves = load_glove(glove_filename)
articles = load_articles(articles_dirname, gloves)


#app.run(host='0.0.0.0', port=0)

# sudo python2 server.py /Users/Ben/data/glove/glove.6B.300d.txt /Users/Ben/data/bbc

# sudo python2 server.py /Users/Ben/data/glove/glove.6B.50d.txt /Users/Ben/data/bbc

#gunicorn -D --threads 4 -b 0.0.0.0:5000 --access-logfile server.log --timeout 60 server:app /Users/Ben/data/glove/glove.6B.300d.txt /Users/Ben/data/bbc


#ssh -i /Users/Ben/Dropbox/License/bliu36msan.pem ec2-user@35.164.160.73

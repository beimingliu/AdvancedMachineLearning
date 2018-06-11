import sys
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def loadkeys(filename):
    """"
    load twitter api keys/tokens from CSV file with form
    consumer_key, consumer_secret, access_token, access_token_secret
    """
    with open(filename) as f:
        items = f.readline().strip().split(', ')
        return items


def authenticate(twitter_auth_filename):
    """
    Given a file name containing the Twitter keys and tokens,
    create and return a tweepy API object.
    """
    consumer_key, consumer_secret, \
    access_token, access_token_secret = loadkeys(twitter_auth_filename)

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    return api

def fetch_tweets(api, name):
    """
    Given a tweepy API object and the screen name of the Twitter user,
    create a list of tweets where each tweet is a dictionary with the
    following keys:

       id: tweet ID
       created: tweet creation date
       retweeted: number of retweets
       text: text of the tweet
       hashtags: list of hashtags mentioned in the tweet
       urls: list of URLs mentioned in the tweet
       mentions: list of screen names mentioned in the tweet
       score: the "compound" polarity score from vader's polarity_scores()

    Return a dictionary containing keys-value pairs:

       user: user's screen name
       count: number of tweets
       tweets: list of tweets, each tweet is a dictionary

    For efficiency, create a single Vader SentimentIntensityAnalyzer()
    per call to this function, not per tweet.
    """
    tweets = []
    analyzer = SentimentIntensityAnalyzer()
    for tweet in tweepy.Cursor(api.user_timeline, id=name).items(100):
        tweet_dict = {}
        tweet_dict['id'] = tweet.id
        tweet_dict['created'] = tweet.created_at
        tweet_dict['retweeted'] = tweet.retweet_count
        tweet_dict['text'] = tweet.text
        tweet_dict['hastags'] = [tweet.entities['hashtags'][i]['text'] for i in range(len(tweet.entities['hashtags']))]
        tweet_dict['urls'] = [tweet.entities['urls'][i]['url'] for i in range(len(tweet.entities['urls']))]
        tweet_dict['mentions'] = [tweet.entities['user_mentions'][i]['screen_name'] for i in
                                  range(len(tweet.entities['user_mentions']))]
        tweet_dict['score'] = analyzer.polarity_scores(tweet.text)['compound']

        tweets.append(tweet_dict)

    dictionary = {}
    dictionary['user'] = name
    dictionary['count'] = len(tweets)
    dictionary['tweets'] = tweets

    return dictionary


def fetch_following(api,name):
    """
    Given a tweepy API object and the screen name of the Twitter user,
    return a a list of dictionaries containing the followed user info
    with keys-value pairs:

       name: real name
       screen_name: Twitter screen name
       followers: number of followers
       created: created date (no time info)
       image: the URL of the profile's image

    To collect data: get a list of "friends IDs" then get
    the list of users for each of those.
    """
    id_list = api.friends_ids(name)
    following_list = []

    for ID in id_list:
        person_dict = {}
        acc_info = api.get_user(ID)
        person_dict['name'] = acc_info.name
        person_dict['screen_name'] = acc_info.screen_name
        person_dict['followers'] = acc_info.followers_count
        person_dict['created'] = acc_info.created_at.date()
        person_dict['image'] = acc_info.profile_image_url_https

        following_list.append(person_dict)

    return following_list
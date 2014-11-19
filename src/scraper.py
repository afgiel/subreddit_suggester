import praw
from pprint import pprint
import csv
import os
from unidecode import unidecode

subreddits = [
  "NoStupidQuestions",
  "shortscarystories",
  "Showerthoughts",
  "DebateReligion",
  "confession",
  "relationship_advice",
  "UnsentLetters",
  "self",
  "askphilosophy",
  "ShittyPoetry",
  "AskMen",
  "AskWomen"
]

num_posts = 10

keys = ['created_utc', 'score', 'domain', 'id', 'title', 'author', 'ups', 'downs', 'num_comments', 'permalink', 'selftext', 'link_flair_text', 'over_18', 'thumbnail', 'subreddit_id', 'edited', 'link_flair_css_class', 'author_flair_css_class', 'is_self', 'name', 'url', 'distinguished']

user_agent = "scrapebot by r/Classifier"
r = praw.Reddit(user_agent=user_agent)



for subreddit in subreddits:
	curr_subreddit = r.get_subreddit(subreddit)
	filename = subreddit + '.csv'

	all_posts = []
	for submission in curr_subreddit.get_hot(limit=num_posts):
		curr_post = {}
		curr_post['created_utc'] = submission.created_utc
		curr_post['score'] = submission.score
		curr_post['domain'] = submission.domain
		curr_post['id'] = submission.id
		curr_post['title'] = submission.title
		curr_post['author'] = submission.author
		curr_post['ups'] = submission.ups
		curr_post['downs'] = submission.downs
		curr_post['num_comments'] = submission.num_comments
		curr_post['permalink'] = submission.permalink
		curr_post['selftext'] = unidecode(submission.selftext)
		curr_post['link_flair_text'] = submission.link_flair_text
		curr_post['over_18'] = submission.over_18
		curr_post['thumbnail'] = submission.thumbnail
		curr_post['subreddit_id'] = submission.subreddit_id
		curr_post['edited'] = submission.edited
		curr_post['link_flair_css_class'] = submission.link_flair_css_class
		curr_post['author_flair_css_class'] = submission.author_flair_css_class
		curr_post['is_self'] = submission.is_self
		curr_post['name'] = submission.name
		curr_post['url'] = submission.url
		curr_post['distinguished'] = submission.distinguished

		
		all_posts.append(curr_post)

	#print all_posts	


	if os.path.isfile(filename):
		with open(filename, 'a') as curr_file:
			dict_writer = csv.DictWriter(curr_file, keys)
			dict_writer.writerows(all_posts)

	else:
		with open(filename, 'wb') as curr_file:
			dict_writer = csv.DictWriter(curr_file, keys)
			dict_writer.writeheader()
			dict_writer.writerows(all_posts)






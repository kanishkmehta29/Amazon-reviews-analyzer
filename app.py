import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import random
import time
import re
import unicodedata
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download("stopwords")
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import streamlit as st
from datetime import datetime
import string

punctuation = string.punctuation

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

summary_headers = {"Authorization": "Bearer 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"}

HEADERS = {
    "accept-language": "en-GB,en;q=0.9",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "cookie": "csm-hit=tb:7R3QBMNWE8SYFXDZH8NG+s-XF9JNNT2H1WRDQG8XPN1|1719032135540&t:1719032135540&adb:adblk_no; lc-acbin=en_IN; session-id=262-3584196-0893719; session-id-time=2082787201; session-token=GUa9wnhHx2CXc/5u4+/xGt1K+vSfezkQkGnjBSFU9H3NwPTQWznqaayC0PNB0P1VnxE9zFTpE02ap/foeXzp+g5QV9JfQLavSnBD8+1AJFjaoa8W0600YyYPdlPv6yyZE6c4vEw/MSnMz7GL6AZ+nNOhp0WMm6iqUxCXtkwGAsWqJUpRU4r792pKc1qDugDTfCBZN79fDT4fEUgn+gxm9kI9q/FaAh3f0j6hFRXd7MrCO2fb5RicV0jif5okCxro61ej8p77dW7PZfO2QSzM78/k2py+Qp1+Ke7gqL4zj/Lqxigl0iuKud265FqMLhTuktO4tHmJqVybR1+f6hK+SxXgbIiYUTLePOr163d5aYJjtWczROssJvGWxEsczw7W; i18n-prefs=INR; ubid-acbin=258-0851131-9942451; x-acbin=\"wHLSxB2KChwwnPb6HxWvrQGYE6fY3E9B1@4dViRPaIJ6i?R3AJ7fkKDqiJJj1aMZ\"; at-acbin=Atza|IwEBIA4y3qvs2V7NJj7ZraTh6T_e-aJVbuaZU-sj2tKZpsT4YgkpGMLABjgBHuqJP2CcouOz5T--EDMZebnQN-Hqc70CizSQB2x-oijwQ4c_HVy_oX5T0Gc1aCUhCTSPoZHB5wTtHgVI5LbluQDK8NMJ8Zm80XszMBBUz-rHoIWczUvj7U_lv-Jw7ct_8UhppUjqaOJ2JEvIMNdsHLM7tE5jXodc7-lGwNk7KpibU51tNiGUxA; sess-at-acbin=\"3N5NerSZeKYosEAjuvjxbqtIss2Q/vc8LBpUoutdtU4=\""
}

def get_soup(url):
    r = requests.get(url, headers=HEADERS)
    random_wait = random.random()
    time.sleep(random_wait)
    soup = bs(r.text, "html.parser")
    return soup

# Function to get review page link
def get_amazon_review_link(product_page_url):
    # Send a request to the product page
    response = requests.get(product_page_url, headers=HEADERS)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to load page with status code {response.status_code}")
    
    # Parse the HTML content
    soup = bs(response.content, 'html.parser')
    
    # Find the link to the reviews page
    review_link_tag = soup.find('a', {'data-hook': 'see-all-reviews-link-foot'})
    
    if review_link_tag:
        review_link = review_link_tag.get('href')
        # Construct the full URL if necessary
        if review_link.startswith('/'):
            review_link = 'https://www.amazon.in' + review_link
        return review_link
    else:
        raise Exception("Could not find the review link on the product page.")

def get_reviews(soup):
    reviewlist = []
    reviews = soup.find_all("div", {"data-hook": "review"})
    try:
        for item in reviews:
            review = {
                "product": soup.title.text.replace(
                    "Amazon.com:Customer reviews:", ""
                ).strip(),
                "title": item.find("a", {"data-hook": "review-title"}).text.strip(),
                "date": soup.find("span", {"data-hook": "review-date"}).text.strip(),
                "rating": float(
                    item.find("i", {"data-hook": "review-star-rating"})
                    .text.replace("out of 5 stars", "")
                    .strip()
                ),
                "body": item.find("span", {"data-hook": "review-body"}).text.strip(),
            }
            reviewlist.append(review)
    except:
        pass
    return reviewlist

def return_dt(ex_string):
    date = datetime.strptime(ex_string.split("on ")[1], "%d %B %Y").date()
    return date

def text_cleaning(text):
    text = text.lower()
    stop = stopwords.words("english")
    text = " ".join([word for word in text.split() if word not in stop])
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text

def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = "".join(clean_list)
    return clean_str

def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search("\d", i):
            list_text_new.append(i)
    return "".join(list_text_new)

def remove_accented_chars(text):
    new_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return new_text

def remove_special_characters(text):
    pat = r"[^a-zA-z0-9]"
    return re.sub(pat, " ", text)

def get_polarity(text):
    textblob = TextBlob(str(text.encode("utf-8")))
    pol = textblob.sentiment.polarity
    return pol

def get_subjectivity(text):
    textblob = TextBlob(str(text.encode("utf-8")))
    subj = textblob.sentiment.subjectivity
    return subj

def plot_to_img():
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return img

def query(payload):
    response = requests.post(API_URL, headers=summary_headers, json=payload)
    return response.json()

st.title('Amazon Reviews Analysis')

product_page_url = st.text_input('Enter the Amazon product page URL')

if st.button('Analyze'):
    if product_page_url:
        reviewlist = []
        review_page_url = get_amazon_review_link(product_page_url)
        soup = get_soup(review_page_url)
        reviewlist.extend(get_reviews(soup))

        for x in range(2, 50):
            url = review_page_url.replace(
                "ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
                f"ref=cm_cr_getr_d_paging_btm_next_{x}?ie=UTF8&reviewerType=all_reviews&pageNumber={x}",
            )
            soup = get_soup(url)
            current_length = len(reviewlist)
            reviewlist.extend(get_reviews(soup))
            if len(reviewlist) == current_length:
                break

        df = pd.DataFrame(reviewlist)

        df["product"] = df["product"].apply(lambda x: x.split(":")[-1])
        df["product"] = df["product"].apply(lambda x: x.split(".")[0])
        df["product"] = df["product"].apply(lambda x: x.split("|")[0])
        df["product"] = df["product"].apply(lambda x: x.split(",")[0])
        df["product"] = df["product"].apply(lambda x: x.split("-")[0])
        df["title"] = df["title"].apply(lambda x: x.split("\n")[1])
        df["date"] = df["date"].apply(return_dt)
        df["date"] = pd.to_datetime(df["date"])
        df["title"] = df["title"].astype(str)
        df["body"] = df["body"].astype(str)
        df["body"] = df["body"].apply(punctuation_removal)
        df["body"] = df["body"].apply(drop_numbers)
        df["body"] = df.apply(lambda x: remove_accented_chars(x["body"]), axis=1)
        df["body"] = df.apply(lambda x: remove_special_characters(x["body"]), axis=1)
        df["length"] = df["body"].apply(len)
        df["polarity"] = df["body"].apply(get_polarity)
        df["subjectivity"] = df["body"].apply(get_subjectivity)
        df["char_count"] = df["body"].apply(len)
        df["word_count"] = df["body"].apply(lambda x: len(x.split()))
        df["word_density"] = df["char_count"] / (df["word_count"] + 1)
        df["punctuation_count"] = df["body"].apply(
            lambda x: len("".join(_ for _ in x if _ in punctuation))
        )

        pos_df = df[df.rating > 4.0]
        neg_df = df[df.rating < 3.0]
        pos_body = pos_df["body"].str.cat(sep=". ")
        pos_title = pos_df["title"].str.cat(sep=". ")
        neg_body = neg_df["body"].str.cat(sep=". ")
        neg_title = neg_df["title"].str.cat(sep=". ")
        pos_body = pos_body[0:1024]
        neg_body = neg_body[0:1024]
        pos_title = pos_title[0:1024]
        neg_title = neg_title[0:1024]

        # Data visualization

        # Rating Distribution
        sample_ratings = df.groupby("rating", sort=False).count()
        sample_ratings = sample_ratings.iloc[:, 0:1]
        sample_ratings.columns = ["count"]
        sample_ratings = sample_ratings.reset_index()
        fig, ax = plt.subplots()
        sns.barplot(
            x="rating",
            y="count",
            data=sample_ratings,
            palette=["#E60049", "#0BB4FF", "#50E991", "#FFA300", "#9B19F5"],
            edgecolor="black",
            ax=ax,
        )
        plt.title("Distribution of Product Ratings", y=1.02)
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        ## Visualizing the Most Frequent Words

        cv = CountVectorizer(stop_words="english")
        words = cv.fit_transform(df.body)
        sum_words = words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        frequency = pd.DataFrame(words_freq, columns=["word", "freq"])
        fig, ax = plt.subplots()
        plt.style.use("fivethirtyeight")
        color = plt.cm.ocean(np.linspace(0, 1, 10))
        frequency.head(10).plot(x="word", y="freq", kind="bar", color=color, ax=ax)
        plt.title("Most Frequently Occurring Words - Top 10")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Generate the text for word clouds
        text1 = " ".join(title for title in df[df.rating > 3.0].title)
        text2 = " ".join(title for title in df[df.rating < 3.0].title)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        if text1:
            word_cloud1 = WordCloud(
                collocations=False, background_color="white", width=2048, height=1080
            ).generate(text1)
            axes[0].imshow(word_cloud1, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title("Word Cloud for Positive Reviews")
        else:
            axes[0].axis("off")
            axes[0].text(
                0.5,
                0.5,
                "No words to display",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=20,
            )

        if text2:
            word_cloud2 = WordCloud(
                collocations=False, background_color="white", width=2048, height=1080
            ).generate(text2)
            axes[1].imshow(word_cloud2, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title("Word Cloud for Negative Reviews")
        else:
            axes[1].axis("off")
            axes[1].text(
                0.5,
                0.5,
                "No words to display",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=20,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Sentiment analysis
        Positive_reviews = len(df[df["polarity"] > 0])
        Neutral_reviews = len(df[df["polarity"] == 0])
        Negative_reviews = len(df[df["polarity"] < 0])
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [Positive_reviews, Neutral_reviews, Negative_reviews]
        colors = sns.color_palette("pastel")[0:3]
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "black"},
        )
        ax.set_title("Categorization of Reviews")
        sns.set_style("whitegrid")
        sns.despine(left=True, bottom=True)
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        ## Visualizing Polarity and Subjectivity

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        sns.set(style="whitegrid")
        sns.kdeplot(df["polarity"], ax=axes[0], shade=True, color="b", bw_adjust=0.5)
        axes[0].set_title("Distribution of Polarity")
        axes[0].set_xlabel("Polarity")
        axes[0].set_ylabel("Density")
        sns.kdeplot(df["subjectivity"], ax=axes[1], shade=True, color="r", bw_adjust=0.5)
        axes[1].set_title("Distribution of Subjectivity")
        axes[1].set_xlabel("Subjectivity")
        fig.suptitle("Distribution of Polarity and Subjectivity", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
        plt.close()

        # Summarizing the reviews
        pos_sum = {"summary_text": ""}
        neg_sum = {"summary_text": ""}
        if len(pos_title) >= 30:
            pos_sum = query({
                "inputs": pos_title,
                "parameters": {"min_length": 30, "max_length": 150},
            })[0]
        if len(neg_title) >= 30:
            neg_sum = query({
                "inputs": neg_title,
                "parameters": {"min_length": 30, "max_length": 150},
            })[0]

        st.subheader("Summarized Positive Reviews")
        st.write(pos_sum["summary_text"])

        st.subheader("Summarized Negative Reviews")
        st.write(neg_sum["summary_text"])
    else:
        st.error("Please enter a valid URL")

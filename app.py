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
from urllib.parse import urlparse
from io import BytesIO

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Generative AI library not available. Install with: pip install google-generativeai")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

punctuation = string.punctuation

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    """Initialize Gemini API with user's API key"""
    if not GEMINI_AVAILABLE:
        return None
    
    # Get API key from Streamlit secrets or user input
    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets else None
    
    if not api_key:
        # Ask user for API key if not in secrets
        api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", 
                                       help="Get your free API key from https://makersuite.google.com/app/apikey")
        if not api_key:
            st.sidebar.warning("Please enter your Gemini API key to use summarization feature.")
            return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        return None

HEADERS = {
    "accept-language": "en-GB,en;q=0.9",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "cookie": "rxc=AFG+G522GKHoKRjyW34; csm-hit=tb:5HNAZB5RNEC15Z4VEWHW+s-5HNAZB5RNEC15Z4VEWHW|1757335445106&t:1757335445106&adb:adblk_no; rx=AQDAXLqbY9Goo57WLj607QhD6qg=@AanNvmg=; session-token=Okso/9az4i4vAkcWTG0sID2/kq5v79ccDA1AQ+8VR43F3ffjZiFcqT8bQ+AZsTSt3nzzubOYLV+I040XG2acJ/K3RH1tMg78aEr8RdieIjeq2i9+8pbWAawIjYJwJ26qymrWjwtjvSRKAaLqbCb0SLTrhelrqxbu2sClcWKzApDS1MMbwJTZECaw2if1rYQZ5m0Mg1adxXAXR7YFo0z1raYrpFPB+YsTgHG98eXuw9HQiFss56s8zzg04ozRloe2RW2nToWlb3XUT85II3TnoBtzPcWmW9QL2B9eQFXzy51G4D2Puar6r5srDhEXn/WgBuC1didBjj/ObsIziBMmImmbdIRiSMinWzp441gNHJUlHYVdZp+R/tWRS3r5J8ge; x-acbin=\"aRqxlsSuxVq?I8?j1V0644HN@HVCE8BBsDEAsZDrHNXWW@za5rC6SBk0qycbjZa4\"; i18n-prefs=INR; lc-acbin=en_IN; session-id=262-2176739-4923428; session-id-time=2082787201l; ubid-acbin=259-3011315-1017647; at-acbin=Atza|IwEBIBy022gmKKteptph-roN904Kf-haxKccHP5LV5ukOztA9fUSD_XgU1sdhpl8kPnGk8wzKvouFyJI64kE8w8nkyZmn6omXB0RCBmWGosNKjI1Nq3wsGirQqSoNl5aVG24vDVpvJBdFZTtqLG1XOj2mXuEf5ss9PEABS3qZ82svwVUqKLJx55XZIdv_DhK6RSOWhI4UHzyDiNDOA75pZ_2-ghZO-jeJiLjAmX7QPQ54tqNETtutNSUIFyXuonw-4L4AvQ; sess-at-acbin=\"Qb39PFHAgnuMOOz/ohTwDnv/uxgMJD3jOoo1nVl/fT8=\"; sst-acbin=Sst1|PQGTeIPNTB-Wu-b9KBG6m7qECZlcSrOCDNfa4aLfPiIbI4b4pGrsze01TjLU-_2qk9wSpRVEFjXS4SnVGtFVjeRJVyDADmrCmaG66CO7H9CTI38FoxkDP0Yn_ivTbBB80dMkUlcLSyJy7Byfi6GpcTEvZOVw4lRdtYVf893ilmZeq6rz-uR5B_et6Z39ek2byY3dmzXt3-HlTuV4ortKoQIBmsodrwUJi-MgWUQtsFVKNCkVi_HraSTS4qMiY8q4NVKNK9ps_BaPSUySV5sCVv-9Nv7xjXUi56hKvfE4ybUfcHA"
}

def get_soup(url):
    r = requests.get(url, headers=HEADERS)
    random_wait = random.random()
    time.sleep(random_wait)
    soup = bs(r.text, "html.parser")
    return soup

# Function to get review page link
def get_amazon_review_link(product_page_url: str) -> str:
    # Extract path from the URL
    path = urlparse(product_page_url).path
    
    # ASINs are 10 characters, usually alphanumeric, starting with B
    match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", path)
    
    if not match:
        raise ValueError("Could not extract ASIN from the product URL.")
    
    asin = match.group(1)
    
    # Construct the review URL
    review_url = f"https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
    return review_url

def get_reviews(soup):
    reviewlist = []
    reviews = soup.find_all('li', {'data-hook': 'review'})
    
    for item in reviews:
        try:
            reviewer = item.find('span', {'class': 'a-profile-name'})
            title = item.find('a', {'data-hook': 'review-title'})
            date = item.find('span', {'data-hook': 'review-date'})
            rating = item.find('i', {'data-hook': 'review-star-rating'}) \
                     or item.find('i', {'data-hook': 'cmps-review-star-rating'})
            body = item.find('span', {'data-hook': 'review-body'})
            verified = item.find('span', {'data-hook': 'avp-badge'})
            
            review = {
                'reviewer': reviewer.get_text(strip=True) if reviewer else None,
                'title': title.get_text(strip=True) if title else None,
                'date': date.get_text(strip=True) if date else None,
                'rating': float(rating.get_text(strip=True).replace('out of 5 stars', '').strip()) if rating else None,
                'body': body.get_text(strip=True) if body else None,
                'verified_purchase': True if verified else False
            }
            reviewlist.append(review)
        except Exception as e:
            print(f"Error parsing review: {e}")
            continue
    
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

def summarize_text_with_gemini(text, gemini_model):
    """Summarize text using Gemini API"""
    if not text or len(text.strip()) < 30:
        return "Not enough text to summarize"
    
    if gemini_model is None:
        return "Gemini API not available. Please check your API key."
    
    try:
        # Create a summarization prompt
        prompt = f"""
        Please provide a concise summary of the following product reviews in 2-3 sentences:
        
        Reviews:
        {text[:4000]}  # Limit text length for API
        
        Summary:
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        st.error(f"Error in Gemini summarization: {e}")
        return "Error occurred during summarization"

def summarize_text(text, summarizer):
    """Fallback summarization function for compatibility"""
    return summarize_text_with_gemini(text, summarizer)

# Initialize session state for managing the app's state
if 'page' not in st.session_state:
    st.session_state.page = 'input'

# Function to switch back to input page
def switch_to_input():
    st.session_state.page = 'input'
    st.session_state.product_page_url = None

if st.session_state.page == 'input':
    product_page_url = st.text_input('Enter the Amazon product page URL')
    
    if st.button('Analyze'):
        if product_page_url:
            st.session_state.product_page_url = product_page_url
            st.session_state.page = 'analysis'
            st.rerun()
        else:
            st.error("Please enter a valid URL")

elif st.session_state.page == 'analysis':
    if st.button('Back', on_click=switch_to_input):
        st.stop()  # Stop further execution if 'Back' is clicked

    product_page_url = st.session_state.product_page_url
    with st.spinner('Fetching reviews...'):
        reviewlist = []
        review_page_url = get_amazon_review_link(product_page_url)
        print(review_page_url)
        soup = get_soup(review_page_url)
        print(soup)
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

        print(df[:5])

        df['title'] = df['title'].apply(lambda x: x.split('stars')[1])
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

        # Improved color palette using seaborn
        color_palette = sns.color_palette("Set2", 10)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Use the color palette in the bar plot
        bars = frequency.head(10).plot(
            x="word", y="freq", kind="barh", color=color_palette, ax=ax, legend=False, edgecolor="black"
        )

        # Adding text annotations
        for i, (word, freq) in enumerate(zip(frequency.head(10)['word'], frequency.head(10)['freq'])):
            ax.text(freq, i, f'{freq}', va='center', ha='left', fontsize=10, color='black')

        # Customizing the plot
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Words")
        ax.set_title("Most Frequently Occurring Words - Top 10")
        ax.invert_yaxis()
        ax.grid(False)
        plt.tight_layout()

        # Displaying the plot using streamlit
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
        fig, ax = plt.subplots(figsize=(8, 8))
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
        st.subheader("Review Summaries")
        
        # Initialize Gemini API
        gemini_model = init_gemini()
        
        if gemini_model is None:
            st.info("Gemini API not available. Please add your API key in the sidebar to enable AI summarization.")
        else:
            st.success("Using Gemini AI for advanced summarization")
        
        # Generate summaries
        pos_summary = "No positive reviews to summarize"
        neg_summary = "No negative reviews to summarize"
        
        if len(pos_title) >= 30:
            pos_summary = summarize_text(pos_title, gemini_model)
            
        if len(neg_title) >= 30:
            neg_summary = summarize_text(neg_title, gemini_model)

        st.subheader("📝 Summarized Positive Reviews")
        st.write(pos_summary)

        st.subheader("📝 Summarized Negative Reviews")
        st.write(neg_summary)
    

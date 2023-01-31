import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from vector_engine.utils import vector_search

#### CUSTOM CSS READ IN ######
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#col1, col2, col3 = st.columns(3)
#col1.metric("Temperature", "70 °F", "1.2 °F")
#col2.metric("Wind", "9 mph", "-8%")
#col3.metric("Humidity", "86%", "4%")

#### CUSTOM CSS READ IN ######

@st.cache
def read_data(data="data/Merged_Dataset_Final.csv"):
    """Read the data from local."""
    return pd.read_csv(data)
    
    #df['year'].astype(int)


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="models/faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("Vector-based searches with Sentence Transformers and Faiss")

    # User search
    user_input = st.text_area("Search box", "Semantic Scholar Literature on Computer Science")

    ### TESTS ###
    OpenAccess = data["isOpenAccess"]
    OpenAccess_List = OpenAccess.copy().unique().tolist()[-2:]

    # Filters
    st.sidebar.markdown("**Filters**")
    filter_year = st.sidebar.slider("Publication year", 1990, 2022, (1990, 2022), 1)
    filter_citations = st.sidebar.slider("Citations", 0, 5000, 0)
    filter_references = st.sidebar.slider("References", 0, 5000, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 50, 10)
    filter_openaccess = st.sidebar.multiselect("Open Access Papers", OpenAccess_List)

    # Fetch results
    if user_input:
        # Get paper IDs
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Slice data on year
        frame = data[
            (data.year >= filter_year[0])        
            & (data.year <= filter_year[1])
            & (data.citationCount >= filter_citations)
            & (data.referenceCount >= filter_references)
            #& (data.isOpenAccess >= filter_openaccess)
        ]
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(frame.indexId):
                f = frame[(frame.indexId == id_)]
            else:
                continue

            st.write(filter_openaccess, 
                f"""**{f.iloc[0].title}**  
            **Citations**: {f.iloc[0].citationCount}
            **References**: {f.iloc[0].referenceCount}
            **Publication year**: {f.iloc[0].year}  
            **Abstract**
            {f.iloc[0].abstract}
            """
            )
            
    if not filter_openaccess:
    	filtered_data = data.copy()
    else:
    	filtered_data = data.loc[data["isOpenAccess"].isin(filter_openaccess)]

#st.dataframe(filtered_data, width=1000, height=500)


if __name__ == "__main__":
    main()

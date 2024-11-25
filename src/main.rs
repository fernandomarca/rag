use async_trait::async_trait;
use futures::StreamExt;
use langchain_rust::add_documents;
use langchain_rust::chain::Chain;
use langchain_rust::chain::ConversationalRetrieverChainBuilder;
use langchain_rust::document_loaders::pdf_extract_loader::PdfExtractLoader;
use langchain_rust::document_loaders::Loader;
use langchain_rust::embedding::ollama_embedder::OllamaEmbedder;
use langchain_rust::embedding::Embedder;
use langchain_rust::fmt_message;
use langchain_rust::fmt_template;
use langchain_rust::llm::client::GenerationOptions;
use langchain_rust::llm::client::Ollama;
use langchain_rust::memory::SimpleMemory;
use langchain_rust::message_formatter;
use langchain_rust::prompt::HumanMessagePromptTemplate;
use langchain_rust::prompt_args;
use langchain_rust::schemas::Message;
use langchain_rust::template_jinja2;
use langchain_rust::text_splitter::TextSplitter;
use langchain_rust::text_splitter::TextSplitterError;
use langchain_rust::vectorstore::pgvector::Store;
use langchain_rust::vectorstore::pgvector::StoreBuilder;
use langchain_rust::vectorstore::Retriever;
use langchain_rust::vectorstore::VecStoreOptions;
use langchain_rust::vectorstore::VectorStore;
use text_splitter::TextSplitter as Splitter;
struct MyTextSplitter {}

#[async_trait]
impl TextSplitter for MyTextSplitter {
    async fn split_text(&self, text: &str) -> Result<Vec<String>, TextSplitterError> {
        let result = Splitter::new(2000)
            .chunks(text)
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        Ok(result)
    }
}

async fn embedding(path: &str, store: Store) {
    let ollama = OllamaEmbedder::default().with_model("mxbai-embed-large");
    let loader = PdfExtractLoader::from_path(path).expect("Failed to create PdfExtractLoader");

    let splitter = MyTextSplitter {};

    let documents = loader
        .load_and_split(splitter)
        .await
        .unwrap()
        .map(|d| d.unwrap())
        .collect::<Vec<_>>()
        .await;

    for doc in &documents {
        let _ = add_documents!(store, &documents).await.map_err(|e| {
            println!("Error adding documents: {:?}", e);
        });
    }
}

async fn ask(store: Store, question: &str) {
    let options = GenerationOptions::default().temperature(0.0).num_thread(8);

    let llm = Ollama::default().with_model("llama3.2").with_options(options);

    let prompt= message_formatter![
                    fmt_message!(Message::new_system_message("Você é um assistente útil")),
                    fmt_template!(HumanMessagePromptTemplate::new(
                    template_jinja2!(
                        "Use as seguintes partes do contexto para responder à pergunta no final. 
                        Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
                        {{context}}
                    
                        Pergunta: {{question}}
                        Resposta útil:",
                        "context",
                        "question"
                    )))];

    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(llm)
        .return_source_documents(true)
        .rephrase_question(true)
        .memory(SimpleMemory::new().into())
        .retriever(Retriever::new(store, 10))
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    let input_variables = prompt_args! {
        "question" => question
    };

    let mut stream = chain.stream(input_variables).await.unwrap();
    while let Some(result) = stream.next().await {
        match result {
            Ok(data) => data.to_stdout().unwrap(),
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let ollama = OllamaEmbedder::default().with_model("mxbai-embed-large:latest");

    let store = StoreBuilder::new()
        .embedder(ollama)
        .vstore_options(VecStoreOptions::default())
        .vector_dimensions(1024)
        .collection_name("pops")
        .connection_url("postgresql://postgres:123456@localhost:5432/postgres")
        .build()
        .await
        .unwrap();

    // embedding("./src/pops.pdf", store).await;

    let question = "procure por um documento que fale sobre mapeamento de pneus";
    ask(store, question).await;
}

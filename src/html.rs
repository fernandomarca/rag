use futures::StreamExt;
use langchain_rust::add_documents;
use langchain_rust::chain::Chain;
use langchain_rust::chain::ConversationalRetrieverChainBuilder;
use langchain_rust::document_loaders::pdf_extract_loader::PdfExtractLoader;
use langchain_rust::document_loaders::HtmlLoader;
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
use langchain_rust::url::Url;
use langchain_rust::vectorstore::pgvector::StoreBuilder;
use langchain_rust::vectorstore::Retriever;
use langchain_rust::vectorstore::VecStoreOptions;
use langchain_rust::vectorstore::VectorStore;

#[tokio::main]
async fn main() {
    let path = "./src/fmm.html";
    let html_loader =
        HtmlLoader::from_path(path, Url::parse("https://fmmagalhaes.com.br/").unwrap())
            .expect("Failed to create html loader");

    let documents = html_loader
        .load()
        .await
        .unwrap()
        .map(|d| d.unwrap())
        .collect::<Vec<_>>()
        .await;

    let ollama = OllamaEmbedder::default().with_model("llama3.2");

    let store = StoreBuilder::new()
        .embedder(ollama)
        .vstore_options(VecStoreOptions::default())
        .collection_name("fmm")
        .vector_dimensions(3072)
        .connection_url("postgresql://postgres:123456@localhost:5432/postgres")
        .build()
        .await
        .unwrap();

    let _ = add_documents!(store, &documents).await.map_err(|e| {
        println!("Error adding documents: {:?}", e);
    });

    let options = GenerationOptions::default().temperature(0.0).num_thread(8);

    let llm = Ollama::default()
        .with_model("llama3.2")
        .with_options(options);

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
        .retriever(Retriever::new(store, 3))
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    let input_variables = prompt_args! {
        "question" => "quanto tempo existe a empresa FMM e qual o diferencial dela?",
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

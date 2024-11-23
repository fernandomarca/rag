use langchain_rust::chain::options::ChainCallOptions;
use langchain_rust::chain::Chain;
use langchain_rust::chain::SQLDatabaseChainBuilder;
use langchain_rust::llm::client::GenerationOptions;
use langchain_rust::llm::client::Ollama;
use langchain_rust::llm::Config;
use langchain_rust::llm::OpenAI;
use langchain_rust::llm::OpenAIConfig;
use langchain_rust::tools::postgres::PostgreSQLEngine;
use langchain_rust::tools::SQLDatabaseBuilder;

#[tokio::main]
async fn main() {
    let options = ChainCallOptions::default();
    // let options_ollama = GenerationOptions::default().temperature(0.0).num_thread(16);

    let llm = Ollama::default().with_model("llama3.2");
    // .with_options(options_ollama);

    let engine = PostgreSQLEngine::new("postgresql://postgres:123456@localhost:5432/postgres")
        .await
        .unwrap();

    let db = SQLDatabaseBuilder::new(engine).build().await.unwrap();
    let chain = SQLDatabaseChainBuilder::new()
        .llm(llm)
        .top_k(1)
        .database(db)
        .options(options)
        .build()
        .expect("Failed to build LLMChain");

    let input_variables = chain
        .prompt_builder()
        .query("encontre em transport_demand a cte_number igual a cte123")
        .build();

    println!("Consulta SQL gerada: {:?}", input_variables);

    match chain.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}

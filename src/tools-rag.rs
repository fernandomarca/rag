use async_trait::async_trait;
use langchain_rust::agent::AgentExecutor;
use langchain_rust::agent::ConversationalAgentBuilder;
use langchain_rust::chain::options::ChainCallOptions;
use langchain_rust::chain::Chain;
use langchain_rust::llm::client::GenerationOptions;
use langchain_rust::llm::client::Ollama;
use langchain_rust::memory::SimpleMemory;
use langchain_rust::prompt_args;
use langchain_rust::tools::CommandExecutor;
use langchain_rust::tools::DuckDuckGoSearchResults;
use langchain_rust::tools::SerpApi;
use langchain_rust::tools::Tool;
use serde_json::Value;
use std::error::Error;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    struct Date {}

    #[async_trait]
    impl Tool for Date {
        fn name(&self) -> String {
            "Date".to_string()
        }
        fn description(&self) -> String {
            "Useful when you need to get the date, input is a query".to_string()
        }
        async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
            let input_str = input.as_str().ok_or("Input should be a string")?;
            println!("Input: {:?}", input_str);
            let now = chrono::Local::now();
            Ok(now.format("%d-%m-%Y %H:%M:%S").to_string())
        }
    }

    let options = GenerationOptions::default().temperature(0.0).num_thread(8);

    let llm = Ollama::default()
        .with_model("llama3.2")
        .with_options(options);

    let memory = SimpleMemory::new();
    let serpapi_tool = SerpApi::default();
    let duckduckgo_tool = DuckDuckGoSearchResults::default();
    let tool_calc = Date {};
    let command_executor = CommandExecutor::default();

    let agent = ConversationalAgentBuilder::new()
        .tools(&[
            Arc::new(serpapi_tool),
            // Arc::new(command_executor),
            // Arc::new(tool_calc),
            Arc::new(duckduckgo_tool),
        ])
        .options(
            ChainCallOptions::new()
                .with_temperature(0.0)
                .with_max_tokens(3000),
        )
        .build(llm)
        .unwrap();

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());

    let input_variables = prompt_args! {
        "input" => "Traga uma notícia sobre a economia atual do Brasil. Além da notícia, quero saber o que os especialistas estão dizendo sobre o assunto e não pode deixar de citar a fonte da informação por exemplo de sites como G1, UOL e não pode faltar a data da publicação da notícia. Descarte as notícias com mais de 1 mês de publicação e sem os critérios definidos.",
    };

    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result.replace("\n", " "));
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}

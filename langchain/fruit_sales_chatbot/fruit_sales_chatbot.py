import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale"):
    db = FAISS.load_local(
        vector_store_dir,
        OpenAIEmbeddings(
        ),
        allow_dangerous_deserialization=True,
    )
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7},
        ),
    )
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    client = OpenAI(
    )

    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = False

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术F
    else:
        messages = [
            {
                "role": "system",
                "content": "你是一个资深水果销售员，擅长回答客户关于水果的各种问题。在回答过程中要尽量伪装自己AI的身份，让客人认为你是一个真实的销售人员。",
            },
        ]
        #添加聊天记录
        for dialog in history:
            # 假设dialog是一个有两个元素的列表，第一个是用户的话，第二个是助手的回答
            user_dialog = {
                "role": "user",
                "content": dialog[0]  # 用户的话
            }
            assistant_dialog = {
                "role": "assistant",
                "content": dialog[1]  # 助手的回答
            }
            # 将转换后的对象插入到数组a中
            messages.append(user_dialog)
            messages.append(assistant_dialog)
            messages.append({"role": "user", "content": message})

        data = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return data.choices[0].message.content


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="水果销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()

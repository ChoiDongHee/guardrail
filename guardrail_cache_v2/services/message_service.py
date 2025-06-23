import redis
import threading

class ChatClient:
    def __init__(self, username, channel='chatroom'):
        self.username = username
        self.channel = channel
        self.redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True,password = 'donghee')
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(self.channel)

    def send_message(self):
        print(f"💬 채팅 시작! 채널: {self.channel}")
        while True:
            msg = input()
            if msg.lower() in ('/exit', '/quit'):
                print("🚪 채팅 종료")
                break
            message = f"{self.username}: {msg}"
            self.redis.publish(self.channel, message)

    def receive_messages(self):
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                print(f"\n📨 {message['data']}")
                print("> ", end='', flush=True)

    def run(self):
        recv_thread = threading.Thread(target=self.receive_messages, daemon=True)
        recv_thread.start()
        self.send_message()


if __name__ == '__main__':
    username = input("닉네임을 입력하세요: ")
    client = ChatClient(username)
    client.run()
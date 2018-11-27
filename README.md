# webserver
we use python3

setup:
- install python3 and pip3
- `pip3 install django`
- `pip3 install channels`
- `pip3 install channels-redis`
- `pip3 install asgiref`
- install docker (`brew cask install docker`)

common issues:
- if you get `AttributeError: Module Pip has no attribute 'main'`, you can try 
(`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py` and `python3 get-pip.py`)

# example client

code:
```js
const WebSocket = require('ws');
const ws = new WebSocket('ws://127.0.0.1:8000/app/test-client/');

ws.on('open', function open() {ws.send(JSON.stringify({'input': 'asl blah blah test'}));});
ws.on('message', function incoming(data) {console.log(data);});
```

setup:
- install nodejs
(`sudo apt-get install nodejs` or https://www.taniarascia.com/how-to-install-and-use-node-js-and-npm-mac-and-windows/)
- copy code above into client.js
- `npm install --save ws`
- `nodejs client.js` or `node client.js`

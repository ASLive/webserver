# webserver

example client:
```js
const WebSocket = require('ws');
const ws = new WebSocket('ws://127.0.0.1:8000/app/test-client/');

ws.on('open', function open() {ws.send(JSON.stringify({'input': 'asl blah blah test'}));});
ws.on('message', function incoming(data) {console.log(data);});
```

//package hello
//
//import org.junit.Assert.*
//
//import java.lang.reflect.Type
//import java.util.ArrayList
//import java.util.concurrent.CountDownLatch
//import java.util.concurrent.TimeUnit
//import java.util.concurrent.atomic.AtomicReference
//
//import org.junit.Before
//import org.junit.Test
//import org.junit.runner.RunWith
//
//import org.springframework.boot.test.context.SpringBootTest
//import org.springframework.boot.web.server.LocalServerPort
//import org.springframework.messaging.converter.MappingJackson2MessageConverter
//import org.springframework.messaging.simp.stomp.StompCommand
//import org.springframework.messaging.simp.stomp.StompFrameHandler
//import org.springframework.messaging.simp.stomp.StompHeaders
//import org.springframework.messaging.simp.stomp.StompSession
//import org.springframework.messaging.simp.stomp.StompSessionHandler
//import org.springframework.messaging.simp.stomp.StompSessionHandlerAdapter
//import org.springframework.test.context.junit4.SpringRunner
//import org.springframework.web.socket.WebSocketHttpHeaders
//import org.springframework.web.socket.client.standard.StandardWebSocketClient
//import org.springframework.web.socket.messaging.WebSocketStompClient
//import org.springframework.web.socket.sockjs.client.SockJsClient
//import org.springframework.web.socket.sockjs.client.Transport
//import org.springframework.web.socket.sockjs.client.WebSocketTransport
//
//@RunWith(SpringRunner::class)
//@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
//class GreetingIntegrationTests {
//
//    @LocalServerPort
//    private val port: Int = 0
//
//    private var sockJsClient: SockJsClient? = null
//
//    private var stompClient: WebSocketStompClient? = null
//
//    private val headers = WebSocketHttpHeaders()
//
//    @Before
//    fun setup() {
//        val transports = ArrayList()
//        transports.add(WebSocketTransport(StandardWebSocketClient()))
//        this.sockJsClient = SockJsClient(transports)
//
//        this.stompClient = WebSocketStompClient(sockJsClient)
//        this.stompClient!!.setMessageConverter(MappingJackson2MessageConverter())
//    }
//
//    @Test
//    @Throws(Exception::class)
//    fun getGreeting() {
//
//        val latch = CountDownLatch(1)
//        val failure = AtomicReference()
//
//        val handler = object : TestSessionHandler(failure) {
//
//            @Override
//            fun afterConnected(session: StompSession, connectedHeaders: StompHeaders) {
//                session.subscribe("/topic/greetings", object : StompFrameHandler() {
//                    @Override
//                    fun getPayloadType(headers: StompHeaders): Type {
//                        return Greeting::class.java
//                    }
//
//                    @Override
//                    fun handleFrame(headers: StompHeaders, payload: Object) {
//                        val greeting = payload as Greeting
//                        try {
//                            assertEquals("Hello, Spring!", greeting.getContent())
//                        } catch (t: Throwable) {
//                            failure.set(t)
//                        } finally {
//                            session.disconnect()
//                            latch.countDown()
//                        }
//                    }
//                })
//                try {
//                    session.send("/app/hello", HelloMessage("Spring"))
//                } catch (t: Throwable) {
//                    failure.set(t)
//                    latch.countDown()
//                }
//
//            }
//        }
//
//        this.stompClient!!.connect("ws://localhost:{port}/gs-guide-websocket", this.headers, handler, this.port)
//
//        if (latch.await(3, TimeUnit.SECONDS)) {
//            if (failure.get() != null) {
//                throw AssertionError("", failure.get())
//            }
//        } else {
//            fail("Greeting not received")
//        }
//
//    }
//
//    private inner class TestSessionHandler(private val failure: AtomicReference<Throwable>) : StompSessionHandlerAdapter() {
//
//        @Override
//        fun handleFrame(headers: StompHeaders, payload: Object) {
//            this.failure.set(Exception(headers.toString()))
//        }
//
//        @Override
//        fun handleException(s: StompSession, c: StompCommand, h: StompHeaders, p: ByteArray, ex: Throwable) {
//            this.failure.set(ex)
//        }
//
//        @Override
//        fun handleTransportError(session: StompSession, ex: Throwable) {
//            this.failure.set(ex)
//        }
//    }
//}

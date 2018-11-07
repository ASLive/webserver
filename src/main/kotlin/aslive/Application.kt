package hello

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication

@SpringBootApplication
object Application {

    fun main(args: Array<String>) {
        SpringApplication.run(Application::class.java, args)
    }
}

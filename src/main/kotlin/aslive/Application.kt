package aslive

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
object Application {

    platformStatic fun main(args: Array<String>) {
        runApplication<Application>(*args)
    }
}

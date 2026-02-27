package com.backend.aiverifysnap;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

@SpringBootApplication
@EnableJpaRepositories("com.backend.aiverifysnap.repository")
public class AiVerifySnapApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiVerifySnapApplication.class, args);
    }

    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }

}

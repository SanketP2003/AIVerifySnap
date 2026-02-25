package com.backend.aiverifysnap.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.Primary;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = "spring.main.allow-bean-definition-overriding=true")
@ActiveProfiles("test")
public class DetectionControllerTest {

    @LocalServerPort
    private int port;

    private WebTestClient webTestClient;

    @Autowired
    public void setWebTestClient() {
        this.webTestClient = WebTestClient.bindToServer().baseUrl("http://localhost:" + port).build();
    }

    @TestConfiguration
    static class TestConfig {
        @Bean
        @Primary
        public WebClient.Builder webClientBuilder() {
            WebClient mockWebClient = mock(WebClient.class);
            WebClient.RequestBodyUriSpec requestBodyUriSpec = mock(WebClient.RequestBodyUriSpec.class);
            WebClient.RequestBodySpec requestBodySpec = mock(WebClient.RequestBodySpec.class);
            WebClient.ResponseSpec responseSpec = mock(WebClient.ResponseSpec.class);

            when(mockWebClient.post()).thenReturn(requestBodyUriSpec);
            when(requestBodyUriSpec.uri("/predict")).thenReturn(requestBodySpec);
            when(requestBodySpec.contentType(any(MediaType.class))).thenReturn(requestBodySpec);
            // Updated to handle the specific body type in the controller
            when(requestBodySpec.body(any(org.springframework.web.reactive.function.BodyInserter.class))).thenReturn(requestBodySpec);
            when(requestBodySpec.retrieve()).thenReturn(responseSpec);
            when(responseSpec.bodyToMono(String.class)).thenReturn(Mono.just("Mocked detection result"));

            WebClient.Builder builder = mock(WebClient.Builder.class);
            when(builder.baseUrl(any(String.class))).thenReturn(builder);
            when(builder.build()).thenReturn(mockWebClient);
            return builder;
        }
    }

    @Test
    public void testDetect() {
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", "test-image-content".getBytes())
                .header("Content-Disposition", "form-data; name=file; filename=test.jpg");

        webTestClient.post()
                .uri("/api/detect")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .bodyValue(bodyBuilder.build())
                .exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("Mocked detection result");
    }
}

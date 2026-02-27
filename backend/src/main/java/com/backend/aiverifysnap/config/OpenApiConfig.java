package com.backend.aiverifysnap.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class OpenApiConfig {

    @Value("${server.port:8080}")
    private String serverPort;

    @Bean
    public OpenAPI customOpenAPI() {
        Server devServer = new Server()
                .url("http://localhost:" + serverPort)
                .description("Development server");

        Server prodServer = new Server()
                .url("https://api.aiverifysnap.com")
                .description("Production server");

        Contact contact = new Contact()
                .name("AI Verify Snap Team")
                .email("support@aiverifysnap.com")
                .url("https://aiverifysnap.com");

        License license = new License()
                .name("MIT License")
                .url("https://opensource.org/licenses/MIT");

        Info info = new Info()
                .title("AI Verify Snap API")
                .description("API for AI Verify Snap application - AI-powered image/content verification and detection")
                .version("1.0.0")
                .contact(contact)
                .license(license);

        return new OpenAPI()
                .info(info)
                .servers(List.of(devServer, prodServer));
    }
}

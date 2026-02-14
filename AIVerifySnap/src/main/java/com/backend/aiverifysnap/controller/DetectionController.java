package com.backend.aiverifysnap.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import reactor.core.publisher.Mono;

import java.util.Map;

@RestController
@RequestMapping("/api/detect")
@Tag(name = "Detection", description = "AI Detection API for image/content verification")
public class DetectionController {

    private final WebClient webClient;

    public DetectionController(
            WebClient.Builder builder,
            @Value("${ml.service.url:http://localhost:8000}") String mlServiceUrl
    ) {
        this.webClient = builder
                .baseUrl(mlServiceUrl)
                .build();
    }

    @Operation(
            summary = "Detect AI-generated content",
            description = "Upload a file to detect if it contains AI-generated content"
    )
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Detection completed successfully",
                    content = @Content(mediaType = "application/json", schema = @Schema(implementation = String.class))),
            @ApiResponse(responseCode = "400", description = "Invalid file provided"),
            @ApiResponse(responseCode = "503", description = "ML Detection service is unavailable"),
            @ApiResponse(responseCode = "500", description = "Internal server error")
    })
    @PostMapping(
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    public Mono<ResponseEntity<Object>> detect(
            @Parameter(description = "File to analyze for AI-generated content", required = true)
            @RequestPart("file") MultipartFile file
    ) {
        return webClient.post()
                .uri("/predict")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(
                        "file", file.getResource()
                ))
                .retrieve()
                .bodyToMono(String.class)
                .map(response -> ResponseEntity.ok((Object) response))
                .onErrorResume(WebClientRequestException.class, ex ->
                    Mono.just(ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                            .body(Map.of(
                                    "error", "ML Detection service is unavailable",
                                    "message", "The AI detection service at the configured URL is not running. Please ensure the ML service is started.",
                                    "details", ex.getMessage()
                            )))
                )
                .onErrorResume(Exception.class, ex ->
                    Mono.just(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                            .body(Map.of(
                                    "error", "Detection failed",
                                    "message", ex.getMessage()
                            )))
                );
    }
}

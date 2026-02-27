package com.backend.aiverifysnap.service;

import com.backend.aiverifysnap.model.DetectionHistory;
import com.backend.aiverifysnap.model.Users;
import com.backend.aiverifysnap.repository.DetectionRepository;
import com.backend.aiverifysnap.repository.UserRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class DetectionService {

    private static final Logger log = LoggerFactory.getLogger(DetectionService.class);

    private final WebClient webClient;
    private final int timeout;
    private final DetectionRepository detectionRepository;
    private final UserRepository userRepository;
    private final ObjectMapper objectMapper;
    private final String mlServiceUrl; // store configured ML service base URL for logging and retries

    public DetectionService(WebClient.Builder builder,
            @Value("${ml.service.url}") String url,
            @Value("${ml.service.timeout:30}") int timeout,
            DetectionRepository detectionRepository,
            UserRepository userRepository) {
        this.webClient = builder.baseUrl(url).build();
        this.timeout = timeout;
        this.detectionRepository = detectionRepository;
        this.userRepository = userRepository;
        this.objectMapper = new ObjectMapper();
        this.mlServiceUrl = url;
        log.info("Configured ML service URL: {}", this.mlServiceUrl);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> detectDeepfake(MultipartFile file) {
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", file.getResource());

        try {
            // Prepare to call /detect (retry with trailing slash if not found)
            String primaryPath = "/detect";
            log.debug("Calling ML service at {}{}", mlServiceUrl, primaryPath);
            Map<String, Object> response = callPredictEndpoint(primaryPath, bodyBuilder);
            if (response != null)
                return response;

            // If we get here, try a trailing-slash variant as a fallback
            String altPath = "/detect/";
            log.debug("Retrying ML service at {}{}", mlServiceUrl, altPath);
            response = callPredictEndpoint(altPath, bodyBuilder);

            return response != null ? response : Map.of("error", "Empty response from ML service");
        } catch (WebClientResponseException wcre) {
            // Log the status and response body for debugging
            String body = null;
            try {
                body = wcre.getResponseBodyAsString();
            } catch (Exception ignored) {
            }
            log.warn("ML service returned status {} for /detect. body={}", wcre.getStatusCode(), body);

            // Return structured error map instead of throwing
            Map<String, Object> err = new HashMap<>();
            err.put("error", "ML service returned error status: " + wcre.getStatusCode());
            if (body != null && !body.isBlank())
                err.put("details", body);
            if (wcre.getStatusCode() == HttpStatus.NOT_FOUND) {
                err.put("hint",
                        "ML endpoint '/detect' not found on the configured ML service. Verify the model server and endpoint path.");
            }
            return err;
        } catch (Exception e) {
            log.error("Failed to call ML service /detect: {}", e.toString());
            Map<String, Object> err = new HashMap<>();
            err.put("error", "Failed to call ML service: " + e.getMessage());
            return err;
        }
    }

    // Helper that invokes the configured ML service path and returns the parsed Map
    // response or null
    private Map<String, Object> callPredictEndpoint(String path, MultipartBodyBuilder bodyBuilder) {
        return webClient.post()
                .uri(path)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(bodyBuilder.build()))
                .retrieve()
                .onStatus(status -> status.is4xxClientError(),
                        clientResponse -> clientResponse.bodyToMono(String.class)
                                .flatMap(body -> reactor.core.publisher.Mono
                                        .error(new RuntimeException("ML service 4xx: " + body))))
                .onStatus(status -> status.is5xxServerError(),
                        clientResponse -> clientResponse.bodyToMono(String.class)
                                .flatMap(body -> reactor.core.publisher.Mono
                                        .error(new RuntimeException("ML service 5xx: " + body))))
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {
                })
                .timeout(Duration.ofSeconds(timeout))
                .block();
    }

    public DetectionHistory saveDetection(Map<String, Object> result, String imagePath, Long userId) {
        DetectionHistory history = new DetectionHistory();
        history.setImagePath(imagePath);
        history.setResultLabel(result.getOrDefault("label", "unknown").toString());

        Object confidence = result.get("confidence");
        if (confidence instanceof Number) {
            history.setConfidenceScore(((Number) confidence).doubleValue());
        }

        history.setAnalysisMetadata(toJson(result));
        history.setScanTimestamp(LocalDateTime.now());

        if (userId != null) {
            Users user = userRepository.findById(userId).orElse(null);
            history.setUser(user);
        }

        return detectionRepository.save(history);
    }

    public List<DetectionHistory> getAllDetections() {
        return detectionRepository.findAll();
    }

    public DetectionHistory getDetectionById(Long scanId) {
        return detectionRepository.findById(scanId)
                .orElseThrow(() -> new RuntimeException("Detection not found with id: " + scanId));
    }

    private String toJson(Map<String, Object> map) {
        try {
            return objectMapper.writeValueAsString(map);
        } catch (Exception e) {
            return map.toString();
        }
    }
}

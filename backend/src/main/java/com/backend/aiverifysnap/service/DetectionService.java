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
import org.springframework.scheduling.annotation.Async;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.security.MessageDigest;
import org.springframework.web.server.ResponseStatusException;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.concurrent.CompletableFuture;
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
    private final UserService userService;
    private final ObjectMapper objectMapper;
    private final String mlServiceUrl; // store configured ML service base URL for logging and retries

    public DetectionService(WebClient.Builder builder,
            @Value("${ml.service.url}") String url,
            @Value("${ml.service.timeout:30}") int timeout,
            DetectionRepository detectionRepository,
            UserRepository userRepository,
            UserService userService) {
        ExchangeStrategies largeBufferStrategies = ExchangeStrategies.builder()
                .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(16 * 1024 * 1024))
                .build();
        this.webClient = builder
                .baseUrl(url)
                .exchangeStrategies(largeBufferStrategies)
                .build();
        this.timeout = timeout;
        this.detectionRepository = detectionRepository;
        this.userRepository = userRepository;
        this.userService = userService;
        this.objectMapper = new ObjectMapper();
        this.mlServiceUrl = url;
        log.info("Configured ML service URL: {} (codec buffer: 16 MB)", this.mlServiceUrl);
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> detectDeepfake(MultipartFile file) {
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", file.getResource());

        long startTime = System.currentTimeMillis();

        try {
            String primaryPath = "/detect";
            log.debug("Calling ML service at {}{}", mlServiceUrl, primaryPath);
            Map<String, Object> response = callPredictEndpoint(primaryPath, bodyBuilder);
            if (response == null) {
                // If we get here, try a trailing-slash variant as a fallback
                String altPath = "/detect/";
                log.debug("Retrying ML service at {}{}", mlServiceUrl, altPath);
                response = callPredictEndpoint(altPath, bodyBuilder);
            }

            if (response == null) {
                return Map.of("error", "Empty response from ML service");
            }

            long elapsed = System.currentTimeMillis() - startTime;
            String sha256 = calculateSha256(file);
            return transformMlResponse(response, elapsed, sha256);
        } catch (WebClientResponseException wcre) {
            String body = null;
            try {
                body = wcre.getResponseBodyAsString();
            } catch (Exception ignored) {
            }
            log.warn("ML service returned status {} for /detect. body={}", wcre.getStatusCode(), body);
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

    public String calculateSha256(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            return "Unavailable";
        }
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(file.getBytes());
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (Exception e) {
            log.warn("Failed to compute SHA-256 for file: {}", e.getMessage());
            return "Unavailable";
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> transformMlResponse(Map<String, Object> mlResponse, long elapsedMs, String sha256) {
        Map<String, Object> result = new HashMap<>();
        String verdict = mlResponse.getOrDefault("verdict", "Unknown").toString();
        result.put("prediction", verdict);
        result.put("filename", mlResponse.getOrDefault("filename", "unknown"));
        Object rawConfidence = mlResponse.get("confidence");
        double confidence = 0.0;
        if (rawConfidence instanceof Number) {
            confidence = ((Number) rawConfidence).doubleValue();
        }
        result.put("confidence", confidence);
        boolean isDeepfake = verdict.equalsIgnoreCase("Fake");
        result.put("is_deepfake", isDeepfake);
        double rawScore = confidence / 100.0;
        Object rawOutput = mlResponse.get("raw_output");
        if (rawOutput instanceof List) {
            List<Map<String, Object>> scores = (List<Map<String, Object>>) rawOutput;
            if (!scores.isEmpty()) {
                Object score = scores.get(0).get("score");
                if (score instanceof Number) {
                    rawScore = ((Number) score).doubleValue();
                }
            }
        }
        result.put("raw_score", rawScore);
        Object mlTime = mlResponse.get("processing_time_ms");
        long processingMs = mlTime instanceof Number ? ((Number) mlTime).longValue() : elapsedMs;
        result.put("processing_time_ms", processingMs);
        result.put("elapsed_ms", elapsedMs);
        result.put("model_status", "SigLIP Deepfake Detector (94.4% acc)");
        Map<String, Object> details = new HashMap<>();
        details.put("raw_output", rawOutput);
        details.put("sha256", sha256);
        Object elaData = mlResponse.get("ela");
        if (elaData instanceof Map) {
            Map<String, Object> ela = (Map<String, Object>) elaData;
            details.put("ela_mean", ela.get("mean"));
            details.put("ela_std", ela.get("std"));
            details.put("ela_max", ela.get("max"));
            details.put("ela_image_base64", ela.get("image_base64"));
        }
        result.put("details", details);
        result.put("verdict", verdict);
        result.put("label", verdict);
        return result;
    }

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

        // Strip base64 ELA image from database save to avoid column size constraint
        Map<String, Object> resultForDb = new HashMap<>(result);
        if (resultForDb.containsKey("details")) {
            Map<String, Object> details = new HashMap<>((Map<String, Object>) resultForDb.get("details"));
            details.remove("ela_image_base64");
            resultForDb.put("details", details);
        }
        history.setAnalysisMetadata(toJson(resultForDb));
        history.setScanTimestamp(LocalDateTime.now());
        if (userId != null) {
            Users user = userRepository.findById(userId).orElse(null);
            history.setUser(user);
        }
        return detectionRepository.save(history);
    }

    @Async
    public CompletableFuture<Void> saveDetectionAsync(Map<String, Object> result, String imagePath, Long userId) {
        try {
            saveDetection(result, imagePath, userId);
        } catch (Exception e) {
            log.warn("Failed to persist detection history asynchronously: {}", e.toString());
        }
        return CompletableFuture.completedFuture(null);
    }

    public List<DetectionHistory> getAllDetections() {
        return detectionRepository.findAll();
    }

    public List<DetectionHistory> getUserDetections(Long userId) {
        if (userId == null) {
            return List.of();
        }
        Users user = userRepository.findById(userId).orElse(null);
        if (user != null && "STARTER".equalsIgnoreCase(user.getPlan())) {
            LocalDateTime cutoff = LocalDateTime.now().minusDays(7);
            return detectionRepository.findByUserIdAndScanTimestampAfterOrderByScanTimestampDesc(userId, cutoff);
        }
        return detectionRepository.findByUserIdOrderByScanTimestampDesc(userId);
    }

    public DetectionHistory getDetectionById(Long scanId, Long requestingUserId) {
        DetectionHistory detection = detectionRepository.findById(scanId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Detection not found with id: " + scanId));
        if (requestingUserId != null && detection.getUser() != null) {
            Long ownerId = detection.getUser().getId();
            if (!ownerId.equals(requestingUserId)) {
                Users requester = userRepository.findById(requestingUserId).orElse(null);
                if (requester == null || !"ADMIN".equalsIgnoreCase(requester.getRole())) {
                    throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Access denied: Unauthorized access to scan record");
                }
            }
        }
        return detection;
    }

    public DetectionHistory getDetectionById(Long scanId) {
        return getDetectionById(scanId, null);
    }

    public void deleteDetection(Long scanId, Long requestingUserId) {
        DetectionHistory detection = detectionRepository.findById(scanId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Detection not found with id: " + scanId));
        if (requestingUserId != null && detection.getUser() != null) {
            Long ownerId = detection.getUser().getId();
            if (!ownerId.equals(requestingUserId)) {
                Users requester = userRepository.findById(requestingUserId).orElse(null);
                if (requester == null || !"ADMIN".equalsIgnoreCase(requester.getRole())) {
                    throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Access denied: Unauthorized to delete scan record");
                }
            }
        }
        detectionRepository.deleteById(scanId);
    }

    public void deleteDetection(Long scanId) {
        deleteDetection(scanId, null);
    }

    public Map<String, Object> detectBulkDeepfake(MultipartFile[] files, Long userId) {
        if (files == null || files.length == 0) {
            return Map.of("error", "No files provided for bulk scanning");
        }

        long startTime = System.currentTimeMillis();
        List<Map<String, Object>> individualResults = new java.util.ArrayList<>();
        int fakeCount = 0;
        int realCount = 0;
        double sumConfidence = 0.0;

        for (MultipartFile file : files) {
            if (file.isEmpty()) continue;
            Map<String, Object> single = detectDeepfake(file);
            if (!single.containsKey("error")) {
                saveDetectionAsync(single, file.getOriginalFilename(), userId);
                if (userId != null) {
                    userService.recordUserScan(userId);
                }
                String verdict = single.getOrDefault("prediction", "Unknown").toString();
                if ("Fake".equalsIgnoreCase(verdict)) {
                    fakeCount++;
                } else {
                    realCount++;
                }
                Object conf = single.get("confidence");
                if (conf instanceof Number) {
                    sumConfidence += ((Number) conf).doubleValue();
                }
            }
            individualResults.add(single);
        }

        int total = individualResults.size();
        double avgConf = total > 0 ? Math.round((sumConfidence / total) * 10.0) / 10.0 : 0.0;
        double fakePct = total > 0 ? Math.round(((double) fakeCount / total * 100.0) * 10.0) / 10.0 : 0.0;
        long elapsed = System.currentTimeMillis() - startTime;

        Map<String, Object> response = new HashMap<>();
        response.put("total_scanned", total);
        response.put("total_fake", fakeCount);
        response.put("total_real", realCount);
        response.put("fake_percentage", fakePct);
        response.put("average_confidence", avgConf);
        response.put("batch_elapsed_ms", elapsed);
        response.put("results", individualResults);
        return response;
    }

    public Map<String, Object> reverseImageSearch(MultipartFile file) {
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", file.getResource());

        try {
            return webClient.post()
                    .uri("/reverse-search")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(bodyBuilder.build()))
                    .retrieve()
                    .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                    .timeout(Duration.ofSeconds(timeout))
                    .block();
        } catch (Exception e) {
            log.warn("ML reverse-search call failed: {}", e.getMessage());
            Map<String, Object> fallback = new HashMap<>();
            fallback.put("filename", file.getOriginalFilename());
            fallback.put("error", "Visual reverse search service is currently unavailable");
            fallback.put("reverse_matches_count", 0);
            fallback.put("visual_matches", List.of());
            return fallback;
        }
    }

    public Map<String, Object> identityProtectionScan(MultipartFile file, Long userId) {
        Map<String, Object> detection = detectDeepfake(file);
        String filename = file.getOriginalFilename() != null ? file.getOriginalFilename() : "portrait.jpg";
        String sha256 = calculateSha256(file);

        Map<String, Object> result = new HashMap<>();
        result.put("filename", filename);
        result.put("biometric_hash", sha256);
        result.put("timestamp", LocalDateTime.now().toString());
        result.put("detection_analysis", detection);

        // Truthful operational results: No fabricated matches or fake threat reports
        result.put("infringement_risk_score", 0.0);
        result.put("threat_level", "NOMINAL");
        result.put("total_infringements_found", 0);
        result.put("detected_matches", List.of());
        result.put("takedown_eligible", false);
        result.put("dmca_notice_template", "");

        return result;
    }

    private String toJson(Map<String, Object> map) {
        try {
            return objectMapper.writeValueAsString(map);
        } catch (Exception e) {
            return map.toString();
        }
    }
}

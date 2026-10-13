package com.backend.aiverifysnap.service;

import com.backend.aiverifysnap.dto.UserDto;
import com.backend.aiverifysnap.model.Users;
import com.backend.aiverifysnap.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public UserDto getUserByName(String name) {
        Users user = userRepository.findByName(name).orElseThrow(() -> new RuntimeException("User not found!"));
        return convertToDto(user);
    }

    private boolean isAdminEmail(String email) {
        return email != null && "2019sanketpatil@gmail.com".equalsIgnoreCase(email.trim());
    }

    public Users registerUser(Users user) {
        if (isAdminEmail(user.getEmail())) {
            user.setRole("ADMIN");
            user.setPlan("ENTERPRISE");
        } else {
            if (user.getRole() == null || user.getRole().isBlank()) {
                user.setRole("USER");
            }
            if (user.getPlan() == null || user.getPlan().isBlank()) {
                user.setPlan("STARTER");
            }
        }
        if (user.getScansToday() == null) {
            user.setScansToday(0);
        }
        return userRepository.save(user);
    }

    @Transactional
    public UserDto updateUser(String name, Users user) {
        Users existingUser = userRepository.findByName(name).orElseThrow(() -> new RuntimeException("User not found"));
        if (user.getName() != null && !user.getName().isEmpty())
            existingUser.setName(user.getName());
        if (user.getEmail() != null && !user.getEmail().isEmpty())
            existingUser.setEmail(user.getEmail());
        if (user.getPasswordHash() != null && !user.getPasswordHash().isEmpty())
            existingUser.setPasswordHash(user.getPasswordHash());
        if (user.getRole() != null)
            existingUser.setRole(user.getRole());
        if (user.getPlan() != null)
            existingUser.setPlan(user.getPlan().toUpperCase());
        if (user.getWebhookUrl() != null)
            existingUser.setWebhookUrl(user.getWebhookUrl());

        Users updatedUser = userRepository.save(existingUser);
        return convertToDto(updatedUser);
    }

    @Transactional
    public UserDto updateUserPlan(String name, String newPlan) {
        Users user = userRepository.findByName(name)
                .orElseGet(() -> userRepository.findByClerkId(name)
                        .orElseThrow(() -> new RuntimeException("User not found: " + name)));
        user.setPlan(newPlan.toUpperCase());
        Users saved = userRepository.save(user);
        return convertToDto(saved);
    }

    @Transactional
    public Map<String, Object> getUserQuota(String identifier) {
        Users user = userRepository.findByName(identifier)
                .orElseGet(() -> userRepository.findByClerkId(identifier).orElse(null));

        Map<String, Object> quota = new HashMap<>();
        String plan = user != null && user.getPlan() != null ? user.getPlan().toUpperCase() : "STARTER";
        quota.put("plan", plan);

        LocalDate today = LocalDate.now();
        int scansToday = 0;
        if (user != null) {
            if (user.getLastScanDate() == null || !user.getLastScanDate().equals(today)) {
                user.setLastScanDate(today);
                user.setScansToday(0);
                userRepository.save(user);
            } else {
                scansToday = user.getScansToday() != null ? user.getScansToday() : 0;
            }
        }

        quota.put("scansToday", scansToday);
        if ("STARTER".equals(plan)) {
            quota.put("dailyLimit", 10);
            quota.put("remainingToday", Math.max(0, 10 - scansToday));
            quota.put("historyDaysLimit", 7);
            quota.put("bulkLimit", 1);
            quota.put("identityProtectionAllowed", false);
            quota.put("pdfExportAllowed", false);
        } else if ("PROFESSIONAL".equals(plan)) {
            quota.put("dailyLimit", -1); // Unlimited
            quota.put("remainingToday", "Unlimited");
            quota.put("historyDaysLimit", -1); // Full
            quota.put("bulkLimit", 50);
            quota.put("identityProtectionAllowed", true);
            quota.put("pdfExportAllowed", true);
        } else {
            // ENTERPRISE
            quota.put("dailyLimit", -1);
            quota.put("remainingToday", "Unlimited");
            quota.put("historyDaysLimit", -1);
            quota.put("bulkLimit", -1); // Unlimited
            quota.put("identityProtectionAllowed", true);
            quota.put("pdfExportAllowed", true);
            quota.put("apiAccessAllowed", true);
        }

        return quota;
    }

    @Transactional
    public String generateApiKey(String identifier) {
        Users user = userRepository.findByName(identifier)
                .orElseGet(() -> userRepository.findByClerkId(identifier)
                        .orElseThrow(() -> new RuntimeException("User not found: " + identifier)));
        String newKey = "avs_live_" + UUID.randomUUID().toString().replace("-", "");
        user.setApiKey(newKey);
        userRepository.save(user);
        return newKey;
    }

    @Transactional
    public void recordUserScan(Long userId) {
        if (userId == null) return;
        Users user = userRepository.findById(userId).orElse(null);
        if (user == null) return;

        LocalDate today = LocalDate.now();
        if (user.getLastScanDate() == null || !user.getLastScanDate().equals(today)) {
            user.setLastScanDate(today);
            user.setScansToday(1);
        } else {
            user.setScansToday((user.getScansToday() != null ? user.getScansToday() : 0) + 1);
        }
        userRepository.save(user);
    }

    @Transactional
    public void deleteUserByName(String name) {
        if (!userRepository.findByName(name).isPresent()) {
            throw new RuntimeException("User not found!");
        }
        userRepository.deleteByName(name);
    }

    @Transactional
    public UserDto syncClerkUser(String clerkId, String name, String email) {
        Users user = userRepository.findByClerkId(clerkId).orElse(null);
        boolean isAdmin = isAdminEmail(email);
        if (user == null) {
            user = new Users();
            user.setClerkId(clerkId);
            user.setRole(isAdmin ? "ADMIN" : "USER");
            user.setPlan(isAdmin ? "ENTERPRISE" : "STARTER");
            user.setScansToday(0);
            user.setLastScanDate(LocalDate.now());
        } else if (isAdmin) {
            user.setRole("ADMIN");
            user.setPlan("ENTERPRISE");
        }
        if (name != null && !name.isEmpty())
            user.setName(name);
        if (email != null && !email.isEmpty())
            user.setEmail(email);
        Users saved = userRepository.save(user);
        return convertToDto(saved);
    }

    public UserDto convertToDto(Users user) {
        return new UserDto(
                user.getId(),
                user.getName(),
                user.getEmail(),
                user.getRole(),
                user.getClerkId(),
                user.getPlan() != null ? user.getPlan() : "STARTER",
                user.getScansToday() != null ? user.getScansToday() : 0,
                user.getApiKey()
        );
    }
}

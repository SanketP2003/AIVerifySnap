package com.backend.aiverifysnap.controller;

import com.backend.aiverifysnap.dto.UserDto;
import com.backend.aiverifysnap.model.Users;
import com.backend.aiverifysnap.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users")
@Tag(name = "Users", description = "User management API")
public class UserController {

    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @Operation(summary = "Get user by name", description = "Retrieve user details by their name")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "User found",
                    content = @Content(schema = @Schema(implementation = UserDto.class))),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @GetMapping("/{name}")
    public ResponseEntity<UserDto> getUser(
            @Parameter(description = "Name of the user to retrieve") @PathVariable String name) {
        UserDto user = userService.getUserByName(name);
        return ResponseEntity.ok(user);
    }

    @Operation(summary = "Register a new user", description = "Create a new user account")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "User registered successfully",
                    content = @Content(schema = @Schema(implementation = Users.class))),
            @ApiResponse(responseCode = "400", description = "Invalid user data")
    })
    @PostMapping("/register")
    public ResponseEntity<Users> registerUser(@RequestBody Users user) {
        Users usersaved = userService.registerUser(user);
        return ResponseEntity.ok(usersaved);
    }

    @Operation(summary = "Update user", description = "Update an existing user's details")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "User updated successfully",
                    content = @Content(schema = @Schema(implementation = UserDto.class))),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @PutMapping("/update/{name}")
    public ResponseEntity<UserDto> updateUsers(
            @Parameter(description = "Name of the user to update") @PathVariable String name,
            @RequestBody Users user) {
        UserDto usersaved = userService.updateUser(name, user);
        return ResponseEntity.ok(usersaved);
    }

    @Operation(summary = "Delete user", description = "Delete a user by their name")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "User deleted successfully"),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @DeleteMapping("/delete/{name}")
    public ResponseEntity<String> deleteUser(
            @Parameter(description = "Name of the user to delete") @PathVariable String name) {
        userService.deleteUserByName(name);
        return ResponseEntity.ok("User deleted successfully!");
    }
}
